from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm, SwiGLUFFN
from .cell import KalmanGuidedLiquidCell

class LiquidMemoryBlock(nn.Module):
    """Process a historical sequence with dual-timescale liquid memory.

    Recurrently update fast and slow states, correct the resulting
    condition features using PIKO observations, and apply the existing
    residual feed-forward refinement.
    """

    def __init__(self, dim: int, mlp_ratio: float=4.0, proj_drop: float=0.0, init_res_scale: float=0.1, min_tau: float=0.1, fast_tau_max: float=2.0, max_tau: float=10.0, motion_kernel_size: int=3) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm_memory = RMSNorm(dim)
        self.physics_norm = RMSNorm(dim)
        self.cell = KalmanGuidedLiquidCell(dim=dim, min_tau=min_tau, fast_tau_max=fast_tau_max, max_tau=max_tau, motion_kernel_size=motion_kernel_size)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio), drop=proj_drop)
        self.gamma_memory = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))
        self.gamma_mlp = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))
        self.last_stats: Dict[str, torch.Tensor] = {}

    def reset_special_initialization(self) -> None:
        self.cell.reset_special_initialization()

    def forward(self, x: torch.Tensor, physics_tokens: torch.Tensor, grid_size: int, fast_tokens: Optional[torch.Tensor]=None, slow_tokens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f'x must be [B,T,N,D], got {tuple(x.shape)}')
        if physics_tokens.shape != x.shape:
            raise ValueError(f'physics token shape must match x: {tuple(physics_tokens.shape)} != {tuple(x.shape)}')
        if fast_tokens is not None and fast_tokens.shape != x.shape:
            raise ValueError('fast_tokens must match x when provided.')
        if slow_tokens is not None and slow_tokens.shape != x.shape:
            raise ValueError('slow_tokens must match x when provided.')
        b, t, n, d = x.shape
        if d != self.dim:
            raise ValueError(f'hidden size mismatch: got {d}, expected {self.dim}')
        if n != int(grid_size) * int(grid_size):
            raise ValueError(f'token count {n} does not match grid_size={grid_size}.')
        x_norm = self.norm_memory(x)
        physics_norm = self.physics_norm(physics_tokens)
        fast_norm = self.norm_memory(fast_tokens) if fast_tokens is not None else x_norm
        slow_norm = self.norm_memory(slow_tokens) if slow_tokens is not None else x_norm
        memory_h = torch.zeros((b, n, d), device=x.device, dtype=torch.float32)
        c_fast = torch.zeros_like(memory_h)
        c_slow = torch.zeros_like(memory_h)
        p_fast = torch.ones((b, n, 1), device=x.device, dtype=torch.float32)
        p_slow = torch.ones((b, n, 1), device=x.device, dtype=torch.float32)
        outputs = []
        stats_accumulator: Dict[str, list[torch.Tensor]] = {}
        final_branch_state: Dict[str, torch.Tensor] = {}
        for frame_idx in range(t):
            x_t = x_norm[:, frame_idx]
            physics_t = physics_norm[:, frame_idx]
            memory_h, corrected_h, c_fast, c_slow, p_fast, p_slow, branch_state, stats = self.cell(x_t=x_t, physics_t=physics_t, h_prev=memory_h, c_fast_prev=c_fast, c_slow_prev=c_slow, p_fast_prev=p_fast, p_slow_prev=p_slow, x_fast_t=fast_norm[:, frame_idx], x_slow_t=slow_norm[:, frame_idx])
            outputs.append(corrected_h)
            final_branch_state = branch_state
            for key, value in stats.items():
                stats_accumulator.setdefault(key, []).append(value)
        h_seq = torch.stack(outputs, dim=1).to(dtype=x.dtype)
        final_branch_state = {key: value.to(dtype=x.dtype) for key, value in final_branch_state.items()}
        self.last_stats = {key: torch.stack(values).mean().detach() for key, values in stats_accumulator.items()}
        return (h_seq, final_branch_state)
