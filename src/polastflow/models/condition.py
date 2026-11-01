from __future__ import annotations
import torch
import torch.nn as nn
from .common.layers import RMSNorm, get_2d_sincos_pos_embed
from .common.patch import BottleneckPatchEmbedding
from .dtlm import DualTimescaleLiquidMemory
from .psr import PolarimetricSpectralRepresentation
from .piko import PhysicsInformedObservationEncoder

class HistoricalConditionEncoder(nn.Module):
    """Construct PolaSTFlow historical conditions from ZH and ZDR sequences.

    PSR produces fused and frequency-dependent polarimetric tokens, PIKO
    encodes observation-derived evolution features, and DTLM combines
    fast/slow liquid memory with Kalman-corrected temporal states.
    The encoder returns fused, fast and slow conditions plus time constants.
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int, history_frames: int, hidden_size: int, depth: int=2, num_heads: int=8, mlp_ratio: float=4.0, bottleneck_dim: int=128, attn_drop: float=0.0, proj_drop: float=0.0) -> None:
        super().__init__()
        self.history_frames = int(history_frames)
        self.hidden_size = int(hidden_size)
        self.frame_patch_embed = BottleneckPatchEmbedding(img_size, patch_size, in_channels, bottleneck_dim, hidden_size)
        num_patches = self.frame_patch_embed.num_patches
        self.grid_size = int(num_patches ** 0.5)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, hidden_size), requires_grad=False)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, history_frames, 1, hidden_size))
        self.spatial_encoder = PolarimetricSpectralRepresentation(img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, band_dim=max(32, bottleneck_dim // 2), proj_drop=proj_drop)
        self.physics_encoder = PhysicsInformedObservationEncoder(img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, bottleneck_dim=bottleneck_dim, mlp_ratio=mlp_ratio, proj_drop=proj_drop)
        self.temporal_encoder = DualTimescaleLiquidMemory(hidden_size=hidden_size, grid_size=self.grid_size, depth=depth, mlp_ratio=mlp_ratio, proj_drop=proj_drop, min_tau=0.1, fast_tau_max=2.0, max_tau=10.0, motion_kernel_size=3)
        self.out_norm = RMSNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos = get_2d_sincos_pos_embed(self.spatial_pos_embed.shape[-1], self.grid_size)
        self.spatial_pos_embed.data.copy_(torch.from_numpy(pos).float().view(1, 1, -1, self.spatial_pos_embed.shape[-1]))
        nn.init.normal_(self.temporal_pos_embed, std=0.02)

    def reset_special_initialization(self) -> None:
        self.temporal_encoder.reset_special_initialization()

    def get_diagnostics(self) -> dict[str, float]:
        stats = self.temporal_encoder.get_diagnostics()
        stats.update(self.spatial_encoder.get_diagnostics())
        return stats

    def forward(self, history: torch.Tensor):
        if history.ndim != 5:
            raise ValueError(f'history must be [B,C,T,H,W], got {tuple(history.shape)}')
        b, c, t, h, w = history.shape
        if t > self.history_frames:
            raise ValueError(f'T={t} exceeds history_frames={self.history_frames}.')
        frame_input = history.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        spatial_tokens, band_tokens = self.spatial_encoder.forward_with_bands(frame_input)
        n, d = (spatial_tokens.shape[1], spatial_tokens.shape[2])
        spatial_tokens = spatial_tokens.view(b, t, n, d)
        spatial_pos = self.spatial_pos_embed.to(dtype=spatial_tokens.dtype, device=spatial_tokens.device)
        temporal_pos = self.temporal_pos_embed[:, :t].to(dtype=spatial_tokens.dtype, device=spatial_tokens.device)
        temporal_tokens = spatial_tokens + spatial_pos + temporal_pos
        band_tokens = band_tokens.view(b, t, 3, n, d)
        low = band_tokens[:, :, 0]
        mid = band_tokens[:, :, 1]
        high = band_tokens[:, :, 2]
        inv_sqrt2 = 2.0 ** (-0.5)
        fast_spatial = (high + mid) * inv_sqrt2
        slow_spatial = (low + mid) * inv_sqrt2
        fast_tokens = fast_spatial + spatial_pos + temporal_pos
        slow_tokens = slow_spatial + spatial_pos + temporal_pos
        physics_tokens = self.physics_encoder.forward_sequence(history)
        cond, fast_cond, slow_cond, tau_fast, tau_slow = self.temporal_encoder(z=temporal_tokens, physics_tokens=physics_tokens, fast_tokens=fast_tokens, slow_tokens=slow_tokens)
        return (self.out_norm(cond), self.out_norm(fast_cond), self.out_norm(slow_cond), tau_fast, tau_slow)
