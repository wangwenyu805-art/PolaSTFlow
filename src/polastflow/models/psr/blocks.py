from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm

class PatchContextBlock(nn.Module):
    """Extract local context from a PSR frequency band on the patch grid.

    Strided patch projection is followed by depthwise spatial mixing and
    pointwise channel refinement with an additive residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int,
        context_kernel: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if context_kernel % 2 == 0:
            raise ValueError("context_kernel must be odd.")
        self.patch_embed = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.norm = nn.GroupNorm(1, dim)
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=dim,
            bias=True,
        )
        self.pw1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.pw2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        residual = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.pw2(self.drop(self.act(self.pw1(x))))
        return residual + x

class ModeRefiner(nn.Module):
    """Refine a common or differential PSR feature mode by local mixing."""

    def __init__(self, dim: int, kernel_size: int = 3, drop: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pw1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.pw2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(self.norm(x))
        x = self.pw2(self.drop(self.act(self.pw1(x))))
        return residual + x
