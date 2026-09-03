from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext101_32x8d

from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed
from .dpt_head import custom_interpolate, _make_scratch, _make_fusion_block


class ResNeXtBackbone(nn.Module):
    """
    Frame-wise ResNeXt101 backbone for local multi-scale features.
    Returns 4 scales:
        f1: 1/4
        f2: 1/8
        f3: 1/16
        f4: 1/32
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        resnet = resnext101_32x8d(weights=None)

        if in_ch != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_ch,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )  # ~1/4
        self.layer2 = resnet.layer2  # ~1/8
        self.layer3 = resnet.layer3  # ~1/16
        self.layer4 = resnet.layer4  # ~1/32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)



class AlbedoDPTHead(nn.Module):
    """
    ResNeXt101-enhanced Albedo DPT head.

    Main idea:
    - token multi-scale features carry global/multi-view consistency
    - ResNeXt101 multi-scale features carry strong local detail
    - fuse them BEFORE DPT scratch fusion (MVInverse style)
    - keep a light refinement after DPT
    - output 3-channel albedo + 1-channel confidence

    Compared with your previous version:
    - more MVInverse-like
    - stronger local features
    - no raw RGB direct final refinement
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,   # 3 albedo + 1 conf
        activation: str = "sigmoid",
        conf_activation: str = "expp1",
        features: int = 256,
        res_in_channels: List[int] = [256, 512, 1024, 2048],
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        down_ratio: int = 1,
        coarse_dim: int = 128,
        shallow_dim: int = 64,
        refine_dim: int = 128,
        use_local_gate: bool = True,
        freeze_backbone_bn: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.output_dim = output_dim
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_local_gate = use_local_gate
        self.freeze_backbone_bn = freeze_backbone_bn

        self.norm = nn.LayerNorm(dim_in)

        # token -> feature projections
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # token feature resize layers
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        # local CNN backbone
        self.local_backbone = ResNeXtBackbone(in_ch=3)

        # project ResNeXt features to token feature dims
        self.fuse_projects = nn.ModuleList(
            [
                nn.Conv2d(res_in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0)
                for i in range(4)
            ]
        )

        if self.use_local_gate:
            self.local_gates = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1), nn.Sigmoid()),
                    nn.Sequential(nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1), nn.Sigmoid()),
                    nn.Sequential(nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1), nn.Sigmoid()),
                    nn.Sequential(nn.Conv2d(out_channels[3], out_channels[3], kernel_size=1), nn.Sigmoid()),
                ]
            )

        # DPT scratch
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        fused_dim = features // 2

        # extra shallow/detail path
        self.shallow_proj = nn.Sequential(
            nn.Conv2d(features, shallow_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.coarse_refine = nn.Sequential(
            nn.Conv2d(fused_dim, coarse_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(coarse_dim, coarse_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(coarse_dim + shallow_dim, refine_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_dim, refine_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.detail_refine = nn.Sequential(
            ResidualRefineBlock(refine_dim),
            ResidualRefineBlock(refine_dim),
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(refine_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=1, stride=1, padding=0),
        )

        if self.freeze_backbone_bn:
            self._freeze_local_backbone_bn()

    def _freeze_local_backbone_bn(self):
        for m in self.local_backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone_bn:
            for m in self.local_backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_preds, chunk_conf = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
            )
            all_preds.append(chunk_preds)
            all_conf.append(chunk_conf)

        return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: Optional[int] = None,
        frames_end_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # local cnn features
        img = images.reshape(B * S, 3, H, W)
        res_features = self.local_backbone(img)   # 4 scales

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1]).contiguous()
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous()
            x = x.reshape(x.shape[0], x.shape[1], patch_h, patch_w).contiguous()
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[dpt_idx](x)

            # project resnext feature
            local_x = self.fuse_projects[dpt_idx](res_features[dpt_idx])

            # Align spatial size (input resolution may not match the /32 pyramid exactly).
            if local_x.shape[-2:] != x.shape[-2:]:
                local_x = custom_interpolate(
                    local_x,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

            if self.use_local_gate:
                gate = self.local_gates[dpt_idx](x)
                local_x = local_x * gate

            x = x + local_x
            out.append(x)
            dpt_idx += 1

        # release reference early
        del res_features

        fused_feat, shallow_feat = self.scratch_forward_with_shallow(out)

        x = custom_interpolate(
            fused_feat,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            x = self._apply_pos_embed(x, W, H)

        x = self.coarse_refine(x)

        shallow_feat = self.shallow_proj(shallow_feat)
        if shallow_feat.shape[-2:] != x.shape[-2:]:
            shallow_feat = custom_interpolate(
                shallow_feat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        x = self.detail_fuse(torch.cat([x, shallow_feat], dim=1))
        x = self.detail_refine(x)

        x = custom_interpolate(
            x,
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )

        x = self.output_head(x)

        preds, conf = activate_head(
            x,
            activation=self.activation,
            conf_activation=self.conf_activation,
        )
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])

        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w,
            patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward_with_shallow(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        out_shallow = layer_1_rn
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out, out_shallow