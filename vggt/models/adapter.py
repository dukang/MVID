import torch
import torch.nn as nn
from typing import Dict, List, Sequence, Tuple, Union


TensorOrList = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]


# =========================================================
# Helpers
# =========================================================

def _clone_structure_as_list(x: TensorOrList) -> List[torch.Tensor]:
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _restore_structure_like(x_ref: TensorOrList, x_list: List[torch.Tensor]) -> TensorOrList:
    if isinstance(x_ref, list):
        return x_list
    if isinstance(x_ref, tuple):
        return tuple(x_list)
    assert len(x_list) == 1
    return x_list[0]


# =========================================================
# Basic blocks
# =========================================================

class PreNormCrossAttention(nn.Module):
    """
    q <- q + CrossAttn(LN(q), LN(kv), LN(kv))
    q <- q + FFN(LN(q))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, return_attn: bool = False):
        q0 = q
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)

        if return_attn:
            out, attn = self.attn(
                qn, kvn, kvn,
                need_weights=True,
                average_attn_weights=False,  # keep per-head weights
            )
        else:
            out, _ = self.attn(qn, kvn, kvn, need_weights=False)
            attn = None

        q = q0 + self.drop(out)
        q = q + self.ffn(q)

        if return_attn:
            return q, attn
        return q


class LearnedQueryReader(nn.Module):
    """
    Learned queries read from patch memory.

    Input:
      memory: [B, N, D]
    Output:
      task_tokens: [B, M, D]
    """
    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        self.blocks = nn.ModuleList([
            PreNormCrossAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

    def forward(self, memory: torch.Tensor, return_attn: bool = False):
        b = memory.shape[0]
        q = self.queries.expand(b, -1, -1)  # [B,M,D]

        attn_list = []
        for blk in self.blocks:
            if return_attn:
                q, attn = blk(q, memory, return_attn=True)
                attn_list.append(attn)  # [B, heads, M, N]
            else:
                q = blk(q, memory)

        if return_attn:
            return q, attn_list
        return q

class DensePatchWriter(nn.Module):
    """
    Use task tokens to write back into dense patch tokens, while keeping shape.

    Input:
      patch_memory: [B, N, D]
      task_tokens:  [B, M, D]

    Output:
      patch_out:    [B, N, D]

    Design:
      patch_out = patch_memory + alpha * writer(patch_memory <- task_tokens)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.writer = PreNormCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, patch_memory: torch.Tensor, task_tokens: torch.Tensor) -> torch.Tensor:
        delta = self.writer(patch_memory, task_tokens) - patch_memory
        patch_out = patch_memory + self.alpha * delta
        return patch_out


class PerLayerPerBranchDenseBlock(nn.Module):
    """
    One DPT layer adapter block.

    For each branch:
      patch_mem -> read task tokens -> write back to patch_mem
      keep special tokens unchanged

    Output:
      x_branch: [B,S,P,D]
    """
    def __init__(
        self,
        dim: int,
        branch_names: Sequence[str] = ("R", "L", "E", "N"),
        num_queries: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
        reader_layers: int = 1,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.branch_names = tuple(branch_names)

        self.readers = nn.ModuleDict({
            name: LearnedQueryReader(
                dim=dim,
                num_queries=num_queries,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                num_layers=reader_layers,
            )
            for name in self.branch_names
        })

        self.writers = nn.ModuleDict({
            name: DensePatchWriter(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                init_scale=init_scale,
            )
            for name in self.branch_names
        })

    def forward(
        self,
        x_layer: torch.Tensor,      # [B,S,P,D]
        patch_start_idx: int,
        return_attn: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        B, S, P, D = x_layer.shape
        assert 0 < patch_start_idx < P, (patch_start_idx, P)

        special_tokens = x_layer[:, :, :patch_start_idx, :]      # [B,S,Ns,D]
        patch_tokens = x_layer[:, :, patch_start_idx:, :]        # [B,S,Np,D]

        Np = patch_tokens.shape[2]
        patch_memory = patch_tokens.reshape(B, S * Np, D).contiguous()       # [B,S*Np,D]

        out = {}
        for name in self.branch_names:
            if return_attn:
                task_tokens, reader_attn = self.readers[name](patch_memory, return_attn=True)
            else:
                task_tokens = self.readers[name](patch_memory)
                reader_attn = None

            patch_out = self.writers[name](patch_memory, task_tokens).contiguous()
            patch_out = patch_out.reshape(B, S, Np, D).contiguous()

            x_branch = torch.cat([special_tokens, patch_out], dim=2).contiguous()

            out[name] = {
                "x_branch": x_branch,
                "task_tokens": task_tokens,
                "patch_out": patch_out,
                "reader_attn": reader_attn,   # list of [B, heads, M, S*Np]
            }

        return out


# =========================================================
# Main multi-layer dense adapter
# =========================================================

class MultiLayerDenseBranchAdapter(nn.Module):
    """
    Query-conditioned dense branch adapter for DPT-compatible multi-layer tokens.

    Goals:
      - only modify the DPT-used layers
      - keep each layer as [B,S,P,D]
      - each head/branch has its own learned queries
      - queries only generate branch-specific dense patch tokens
      - special tokens stay unchanged

    Input:
      aggregated_tokens_list: list/tuple of [B,S,P,D], or single [B,S,P,D]
      patch_start_idx: number of special tokens per frame
    Output:
      {
        "head_tokens_last": {
            "R": [B,S,P,D],
            "L": [B,S,P,D],
            "E": [B,S,P,D],
            "N": [B,S,P,D],
        },
        "head_token_lists": {
            "R": list/tuple matching input structure, but DPT-used layers replaced,
            "L": ...,
            "E": ...,
            "N": ...,
        },
        "aux_for_loss": {
            "task_tokens": {
                "R": {layer_idx: [B,M,D], ...},
                ...
            },
            "writer_alpha": {
                "R": {layer_idx: scalar, ...},
                ...
            }
        }
      }
    """
    def __init__(
        self,
        token_dim: int,
        branch_names: Sequence[str] = ("R", "L", "E", "N"),
        dpt_layer_indices: Sequence[int] = (4, 11, 17, 23),
        num_queries: int = 8,
        adapter_heads: int = 8,
        adapter_dropout: float = 0.0,
        adapter_mlp_ratio: float = 2.0,
        reader_layers: int = 1,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.branch_names = tuple(branch_names)
        self.dpt_layer_indices = tuple(dpt_layer_indices)

        self.layer_blocks = nn.ModuleDict({
            str(layer_idx): PerLayerPerBranchDenseBlock(
                dim=token_dim,
                branch_names=self.branch_names,
                num_queries=num_queries,
                num_heads=adapter_heads,
                dropout=adapter_dropout,
                mlp_ratio=adapter_mlp_ratio,
                reader_layers=reader_layers,
                init_scale=init_scale,
            )
            for layer_idx in self.dpt_layer_indices
        })

    def forward(
        self,
        aggregated_tokens_list: TensorOrList,
        patch_start_idx: int,
        return_attn: bool = False,
    ) -> Dict[str, object]:
        tokens_list = _clone_structure_as_list(aggregated_tokens_list)

        if len(tokens_list) <= max(self.dpt_layer_indices):
            raise ValueError(
                f"aggregated_tokens_list has length {len(tokens_list)}, "
                f"but max dpt_layer_indices is {max(self.dpt_layer_indices)}"
            )

        branch_token_lists = {
            name: list(tokens_list)
            for name in self.branch_names
        }

        aux_task_tokens = {name: {} for name in self.branch_names}
        aux_writer_alpha = {name: {} for name in self.branch_names}
        aux_reader_attn = {name: {} for name in self.branch_names}

        for layer_idx in self.dpt_layer_indices:
            x_layer = tokens_list[layer_idx]  # [B,S,P,D]
            if x_layer.dim() != 4:
                raise ValueError(
                    f"Expected aggregated_tokens_list[{layer_idx}] to be [B,S,P,D], "
                    f"got {tuple(x_layer.shape)}"
                )

            B, S, P, D = x_layer.shape
            if D != self.token_dim:
                raise ValueError(
                    f"Layer {layer_idx} dim mismatch: got {D}, expected {self.token_dim}"
                )

            layer_out = self.layer_blocks[str(layer_idx)](
                x_layer=x_layer,
                patch_start_idx=patch_start_idx,
                return_attn=return_attn,
            )

            for name in self.branch_names:
                branch_token_lists[name][layer_idx] = layer_out[name]["x_branch"]
                aux_task_tokens[name][layer_idx] = layer_out[name]["task_tokens"]
                aux_writer_alpha[name][layer_idx] = self.layer_blocks[str(layer_idx)].writers[name].alpha
                if return_attn:
                    aux_reader_attn[name][layer_idx] = layer_out[name]["reader_attn"]

        last_idx = self.dpt_layer_indices[-1]
        head_tokens_last = {
            name: branch_token_lists[name][last_idx]
            for name in self.branch_names
        }

        head_token_lists = {
            name: _restore_structure_like(aggregated_tokens_list, branch_token_lists[name])
            for name in self.branch_names
        }

        out = {
            "head_tokens_last": head_tokens_last,
            "head_token_lists": head_token_lists,
            "aux_for_loss": {
                "task_tokens": aux_task_tokens,
                "writer_alpha": aux_writer_alpha,
            },
        }

        if return_attn:
            out["vis"] = {
                "reader_attn": aux_reader_attn
            }

        return out
