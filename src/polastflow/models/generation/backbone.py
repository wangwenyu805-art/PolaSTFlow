from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..condition import HistoricalConditionEncoder
from ..common.layers import TimestepEmbedder
from ..common.patch import BottleneckPatchEmbedding
from .blocks import PMFFinalLayer, PMFTransformerBlock
from .rope import build_2d_rope_tables

class PixelMeanFlowSequenceBackbone(nn.Module):
    """Predict a joint future radar sequence conditioned on PolaSTFlow history.

    Future ZH/ZDR frames are represented jointly as trajectory patches.
    Historical PSR-PIKO-DTLM conditions enter the Pixel MeanFlow
    Transformer through cross-attention, and horizon-dependent conditions
    support frame-specific clean-data decoding in the generation heads.
    """

    def __init__(self, img_size: int=256, patch_size: int=16, radar_channels: int=2, history_frames: int=10, future_frames: int=10, hidden_size: int=384, depth: int=8, aux_head_depth: int=2, num_heads: int=8, mlp_ratio: float=8.0 / 3.0, bottleneck_dim: int=128, attn_drop: float=0.0, proj_drop: float=0.0, num_interval_tokens: int=4, history_temporal_depth: int=1, condition_bottleneck_dim: int=96, condition_mlp_ratio: float=4.0) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError('img_size must be divisible by patch_size.')
        if depth <= aux_head_depth:
            raise ValueError('depth must be larger than aux_head_depth.')
        if hidden_size % num_heads != 0:
            raise ValueError('hidden_size must be divisible by num_heads.')
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.radar_channels = int(radar_channels)
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        self.hidden_size = int(hidden_size)
        self.num_interval_tokens = int(num_interval_tokens)
        self.trajectory_channels = self.radar_channels * self.future_frames
        self.future_patch_embed = BottleneckPatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=self.trajectory_channels, pca_dim=bottleneck_dim, embed_dim=hidden_size)
        self.num_patches = self.future_patch_embed.num_patches
        self.grid_size = int(self.num_patches ** 0.5)
        self.history_encoder = HistoricalConditionEncoder(img_size=img_size, patch_size=patch_size, in_channels=self.radar_channels, history_frames=history_frames, hidden_size=hidden_size, depth=history_temporal_depth, num_heads=num_heads, mlp_ratio=condition_mlp_ratio, bottleneck_dim=condition_bottleneck_dim, attn_drop=attn_drop, proj_drop=proj_drop)

        def _init_history_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        self.history_encoder.apply(_init_history_module)
        reset_special = getattr(self.history_encoder, 'reset_special_initialization', None)
        if callable(reset_special):
            reset_special()
        self.interval_embedder = TimestepEmbedder(hidden_size)
        self.interval_tokens = nn.Parameter(torch.randn(1, num_interval_tokens, hidden_size) / hidden_size ** 0.5)
        self.trajectory_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        total_tokens = num_interval_tokens + self.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, hidden_size) * 0.02)
        shared_depth = depth - aux_head_depth
        block_kwargs = dict(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop)
        self.shared_blocks = nn.ModuleList([PMFTransformerBlock(**block_kwargs) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([PMFTransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
        self.v_heads = nn.ModuleList([PMFTransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
        self.u_final_layer = PMFFinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=self.trajectory_channels)
        self.v_final_layer = PMFFinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=self.trajectory_channels)
        head_dim = hidden_size // num_heads
        rope_cos, rope_sin = build_2d_rope_tables(head_dim, self.grid_size)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)
        patch_ids = torch.arange(1, self.num_patches + 1, dtype=torch.long)
        rope_ids = torch.cat([torch.zeros(num_interval_tokens, dtype=torch.long), patch_ids], dim=0)
        self.register_buffer('rope_ids', rope_ids, persistent=False)

    def _flatten_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f'x must be [B,C,T,H,W], got {tuple(x.shape)}')
        b, c, t, h, w = x.shape
        if c != self.radar_channels or t != self.future_frames:
            raise ValueError(f'trajectory shape mismatch: got C={c},T={t}, expected C={self.radar_channels},T={self.future_frames}.')
        return x.permute(0, 2, 1, 3, 4).contiguous().view(b, self.trajectory_channels, h, w)

    def _unpatchify_trajectory(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError('patches must be [B,N,P*P*C*T].')
        b, n, dim = patches.shape
        p = self.patch_size
        g = int(n ** 0.5)
        expected = p * p * self.trajectory_channels
        if g * g != n or dim != expected:
            raise ValueError(f'patch shape mismatch: N={n}, dim={dim}, expected N={self.num_patches}, dim={expected}.')
        x = patches.view(b, g, g, p, p, self.trajectory_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(b, self.trajectory_channels, g * p, g * p)
        x = x.view(b, self.future_frames, self.radar_channels, g * p, g * p)
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def encode_history(self, history: torch.Tensor):
        base_cond, fast_cond, slow_cond, tau_fast, tau_slow = self.history_encoder(history)
        with torch.amp.autocast(device_type=history.device.type, enabled=False):
            step = torch.arange(self.future_frames, device=history.device, dtype=torch.float32).view(1, self.future_frames, 1, 1)
            tau_f = tau_fast.float().unsqueeze(1)
            tau_s = tau_slow.float().unsqueeze(1)
            retain_fast = torch.exp(-step / tau_f)
            grow_slow = 1.0 - torch.exp(-step / tau_s)
            denom = (retain_fast + grow_slow).clamp_min(torch.finfo(torch.float32).eps)
            w_fast = retain_fast / denom
            w_slow = grow_slow / denom
            horizon_cond = w_fast * fast_cond.float().unsqueeze(1) + w_slow * slow_cond.float().unsqueeze(1)
        horizon_cond = horizon_cond.to(dtype=base_cond.dtype)
        return (base_cond, horizon_cond)

    def _build_sequence(self, noisy_future: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
        b = noisy_future.shape[0]
        trajectory = self.future_patch_embed(self._flatten_trajectory(noisy_future))
        h_embed = self.interval_embedder(interval.reshape(-1))
        h_tokens = self.interval_tokens + h_embed.unsqueeze(1)
        trajectory = trajectory + self.trajectory_type_embed
        seq = torch.cat([h_tokens, trajectory], dim=1)
        return seq + self.pos_embed.to(device=seq.device, dtype=seq.dtype)

    def _run_blocks(self, seq: torch.Tensor, blocks: nn.ModuleList, cond: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            seq = block(seq, self.rope_ids, self.rope_cos, self.rope_sin, condition=cond, trajectory_tokens=self.num_patches)
        return seq

    def _decode_head(self, seq: torch.Tensor, head_blocks: nn.ModuleList, final_layer: nn.Module, base_cond: torch.Tensor, horizon_cond: torch.Tensor) -> torch.Tensor:
        seq = self._run_blocks(seq, head_blocks, base_cond)
        image_tokens = seq[:, -self.num_patches:]
        patches = final_layer.forward_horizon(image_tokens, base_condition=base_cond, horizon_condition=horizon_cond, future_frames=self.future_frames, channels_per_frame=self.radar_channels)
        return self._unpatchify_trajectory(patches)

    def forward_with_condition(self, noisy_future: torch.Tensor, interval: torch.Tensor, cond: torch.Tensor, *, need_u: bool=True, need_v: bool=True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not need_u and (not need_v):
            raise ValueError('At least one head must be requested.')
        b = noisy_future.shape[0]
        if not isinstance(cond, tuple) or len(cond) != 2:
            raise ValueError('cond must be (base_condition, horizon_condition).')
        base_cond, horizon_cond = cond
        expected_base = (b, self.num_patches, self.hidden_size)
        expected_horizon = (b, self.future_frames, self.num_patches, self.hidden_size)
        if base_cond.shape != expected_base:
            raise ValueError(f'base condition must be {expected_base}, got {tuple(base_cond.shape)}')
        if horizon_cond.shape != expected_horizon:
            raise ValueError(f'horizon condition must be {expected_horizon}, got {tuple(horizon_cond.shape)}')
        seq = self._build_sequence(noisy_future, interval)
        seq = self._run_blocks(seq, self.shared_blocks, base_cond)
        clean_u = self._decode_head(seq, self.u_heads, self.u_final_layer, base_cond, horizon_cond) if need_u else None
        clean_v = self._decode_head(seq, self.v_heads, self.v_final_layer, base_cond, horizon_cond) if need_v else None
        return (clean_u, clean_v)
