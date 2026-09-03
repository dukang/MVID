# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.albedo_dpt_head import AlbedoDPTHead

from vggt.models.adapter import MultiLayerDenseBranchAdapter


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=False,
        enable_depth=True,
        enable_track=True,
        enable_albedo=True,
        enable_normal=True,
        enable_shading: bool = True,
        enable_residual: bool = True,
        enable_intrinsic_adapter: bool = True,
    ):
        super().__init__()

        token_dim = 2 * embed_dim

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # -------------------------------------------------
        # Standard heads
        # -------------------------------------------------
        self.camera_head = CameraHead(dim_in=token_dim) if enable_camera else None

        self.point_head = (
            DPTHead(
                dim_in=token_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )

        self.depth_head = (
            DPTHead(
                dim_in=token_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )

        self.track_head = TrackHead(dim_in=token_dim, patch_size=patch_size) if enable_track else None

        # -------------------------------------------------
        # Intrinsic heads
        # -------------------------------------------------
        self.albedo_head = (
            AlbedoDPTHead(
                dim_in=token_dim,
                patch_size=patch_size,
                output_dim=4,   # 3 RGB + 1 conf
                activation="sigmoid",
                conf_activation="expp1",
                features=256,
                res_in_channels=[256, 512, 1024, 2048],
                out_channels=[256, 512, 1024, 1024],
                intermediate_layer_idx=[3, 8, 13, 17],
                pos_embed=True,
                down_ratio=1,
                coarse_dim=128,
                shallow_dim=64,
                refine_dim=128,
                use_local_gate=True,
                freeze_backbone_bn=True,
            )
            if enable_albedo
            else None
        )

        self.normal_head = (
            DPTHead(
                dim_in=token_dim,
                output_dim=4,   # 3 normal + 1 conf
                activation="normal",
                conf_activation="expp1",
                intermediate_layer_idx=[3, 8, 13, 17]
            )
            if enable_normal
            else None
        )

        self.shading_head = (
            DPTHead(
                dim_in=token_dim,
                output_dim=4,   # 3 + conf
                activation="shading",
                conf_activation="expp1",
                intermediate_layer_idx=[3, 8, 13, 17]
            )
            if enable_shading
            else None
        )

        self.residual_head = (
            DPTHead(
                dim_in=token_dim,
                output_dim=4,   # 3 + conf
                activation="residual",
                conf_activation="expp1",
                intermediate_layer_idx=[3, 8, 13, 17]

            )
            if enable_residual
            else None
        )

        # -------------------------------------------------
        # Multi-layer dense intrinsic adapter
        # -------------------------------------------------
        self.enable_intrinsic_adapter = enable_intrinsic_adapter
        self.has_intrinsic_heads = any([
            self.albedo_head is not None,
            self.normal_head is not None,
            self.shading_head is not None,
            self.residual_head is not None,
        ])

        if self.enable_intrinsic_adapter and self.has_intrinsic_heads:
            self.intrinsic_adapter = MultiLayerDenseBranchAdapter(
                token_dim=token_dim,
                branch_names=("R", "L", "E", "N"),
                dpt_layer_indices=(3, 8, 13, 17),
                num_queries=8,
                adapter_heads=8,
                adapter_dropout=0.0,
                adapter_mlp_ratio=2.0,
                reader_layers=1,
                init_scale=0.1,
            )
        else:
            self.intrinsic_adapter = None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Args:
            images:
                [S, 3, H, W] or [B, S, 3, H, W], in [0, 1]
            query_points:
                [N, 2] or [B, N, 2], optional

        Returns:
            predictions dict
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)

        if query_points is not None and query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        predictions["patch_start_idx"] = patch_start_idx


        with torch.cuda.amp.autocast(enabled=False):
            # -------------------------------------------------
            # Standard heads read the original shared tokens.
            # -------------------------------------------------
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # -------------------------------------------------
        # Factor-query adapter: produce factor-conditioned dense tokens.
        # -------------------------------------------------
        if self.intrinsic_adapter is not None:
            adapter_out = self.intrinsic_adapter(
                aggregated_tokens_list=aggregated_tokens_list,
                patch_start_idx=patch_start_idx,
            )

            intrinsic_token_lists = adapter_out["head_token_lists"]

            tokens_R = intrinsic_token_lists["R"]
            tokens_L = intrinsic_token_lists["L"]
            tokens_E = intrinsic_token_lists["E"]
            tokens_N = intrinsic_token_lists["N"]

            # expose adapter internals for debug / optional loss
            predictions["adapter_aux_for_loss"] = adapter_out["aux_for_loss"]
            # predictions["adapter_head_tokens_last"] = adapter_out["head_tokens_last"]
            # predictions["adapter_head_token_lists"] = adapter_out["head_token_lists"]

        else:
            # Without the adapter, intrinsic heads read the shared tokens directly.
            tokens_R = aggregated_tokens_list
            tokens_L = aggregated_tokens_list
            tokens_E = aggregated_tokens_list
            tokens_N = aggregated_tokens_list

        if self.normal_head is not None:
            normal_pred, normal_conf = self.normal_head(
                tokens_N,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["normal"] = normal_pred
            predictions["normal_conf"] = normal_conf

        # -------------------------------------------------
        # Intrinsic heads
        # -------------------------------------------------
        if self.albedo_head is not None:
            albedo_pred, albedo_conf = self.albedo_head(
                tokens_R,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["albedo"] = albedo_pred
            predictions["albedo_conf"] = albedo_conf

        if self.shading_head is not None:
            shading_pred, shading_conf = self.shading_head(
                tokens_L,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["shading"] = shading_pred
            predictions["shading_conf"] = shading_conf

        if self.residual_head is not None:
            residual_pred, residual_conf = self.residual_head(
                tokens_E,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["residual"] = residual_pred
            predictions["residual_conf"] = residual_conf

        # -------------------------------------------------
        # The track head reads the original shared tokens.
        # -------------------------------------------------
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions