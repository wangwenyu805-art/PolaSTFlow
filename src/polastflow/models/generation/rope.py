from __future__ import annotations
from typing import Tuple
import math
import torch

def build_2d_rope_tables(head_dim: int, grid_size: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two-dimensional rotary position embeddings for patch tokens."""
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2-D RoPE, got {head_dim}.")

    axis_pairs = head_dim // 4
    freq = 1.0 / (
        theta ** (torch.arange(axis_pairs, dtype=torch.float32) / max(axis_pairs, 1))
    )

    coords = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    angles_y = yy * freq.reshape(1, -1)
    angles_x = xx * freq.reshape(1, -1)
    angles = torch.cat([angles_y, angles_x], dim=-1)  # [N, head_dim/2]

    zeros = torch.zeros(1, head_dim // 2, dtype=torch.float32)
    angles = torch.cat([zeros, angles], dim=0)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x: torch.Tensor, rope_ids: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor) -> torch.Tensor:
    """Apply spatial rotary embeddings to ``[B, heads, tokens, dim]``."""
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,L,Dh], got {tuple(x.shape)}")
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    cos = cos_table.index_select(0, rope_ids).to(device=x.device, dtype=x.dtype)
    sin = sin_table.index_select(0, rope_ids).to(device=x.device, dtype=x.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,Dh/2]
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)
