from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from ..common.layers import RMSNorm
from .block import LiquidMemoryBlock

class DualTimescaleLiquidMemory(nn.Module):
    """Aggregate dual-timescale, physics-corrected temporal features.

    Each LiquidMemoryBlock processes PSR-derived fused, fast and slow
    token streams with PIKO observations. The final historical states
    provide fused, fast and slow condition tokens and adaptive time
    constants for the generation backbone.
    """

    def __init__(self, hidden_size: int, grid_size: int, depth: int=2, mlp_ratio: float=4.0, proj_drop: float=0.0, init_res_scale: float=0.1, min_tau: float=0.1, fast_tau_max: float=2.0, max_tau: float=10.0, motion_kernel_size: int=3) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.grid_size = int(grid_size)
        self.blocks = nn.ModuleList([LiquidMemoryBlock(dim=hidden_size, mlp_ratio=mlp_ratio, proj_drop=proj_drop, init_res_scale=init_res_scale, min_tau=min_tau, fast_tau_max=fast_tau_max, max_tau=max_tau, motion_kernel_size=motion_kernel_size) for _ in range(depth)])
        self.temporal_score = nn.Sequential(RMSNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.out_proj = nn.Sequential(RMSNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.out_norm = RMSNorm(hidden_size)
        self.pool_res_scale = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))
        self.out_res_scale = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))

    def reset_special_initialization(self) -> None:
        for block in self.blocks:
            block.reset_special_initialization()

    def get_diagnostics(self) -> Dict[str, float]:
        if not self.blocks:
            return {}
        return {key: float(value.detach().cpu()) for key, value in self.blocks[-1].last_stats.items()}

    def forward(self, z: torch.Tensor, physics_tokens: torch.Tensor, fast_tokens: Optional[torch.Tensor]=None, slow_tokens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 4:
            raise ValueError(f'z must be [B,T,N,D], got {tuple(z.shape)}')
        if physics_tokens.shape != z.shape:
            raise ValueError(f'physics_tokens must match z: {tuple(physics_tokens.shape)} != {tuple(z.shape)}')
        if fast_tokens is not None and fast_tokens.shape != z.shape:
            raise ValueError('fast_tokens must match z when provided.')
        if slow_tokens is not None and slow_tokens.shape != z.shape:
            raise ValueError('slow_tokens must match z when provided.')
        b, t, n, d = z.shape
        if n != self.grid_size * self.grid_size:
            raise ValueError(f'N={n} does not match grid_size={self.grid_size}.')
        if d != self.hidden_size:
            raise ValueError(f'hidden size mismatch: got {d}, expected {self.hidden_size}.')
        x = z
        final_branch_state: Dict[str, torch.Tensor] = {}
        for block_idx, block in enumerate(self.blocks):
            x, final_branch_state = block(x=x, physics_tokens=physics_tokens, grid_size=self.grid_size, fast_tokens=fast_tokens if block_idx == 0 else None, slow_tokens=slow_tokens if block_idx == 0 else None)
        base_cond = self.out_norm(x[:, -1])
        fast_cond = self.out_norm(final_branch_state['fast'])
        slow_cond = self.out_norm(final_branch_state['slow'])
        tau_fast = final_branch_state['tau_fast']
        tau_slow = final_branch_state['tau_slow']
        return (base_cond, fast_cond, slow_cond, tau_fast, tau_slow)
