from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm, SwiGLUFFN
from .attention import PMFSelfAttention, PMFCrossAttention

class PMFTransformerBlock(nn.Module):
    """Combine trajectory self-attention and historical cross-attention.

    The block updates trajectory and interval tokens with Pixel MeanFlow
    self-attention, injects historical conditions into trajectory tokens,
    and then applies the existing residual feed-forward network.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8.0 / 3.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = PMFSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.cross_attn = PMFCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size,
            int(hidden_size * mlp_ratio),
            drop=proj_drop,
        )

        # Learnable residual scaling for self-attention and feed-forward updates.
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        rope_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        trajectory_tokens: int = 0,
    ) -> torch.Tensor:
        # Update the token sequence with Pixel MeanFlow self-attention.
        attn_raw = self.attn(self.norm1(x), rope_ids, rope_cos, rope_sin)
        x = x + attn_raw * self.attn_scale

        # Cross-attend trajectory queries to historical condition keys and values.
        if condition is not None and trajectory_tokens > 0:
            if trajectory_tokens > x.shape[1]:
                raise ValueError("trajectory_tokens exceeds sequence length.")
            prefix = x[:, :-trajectory_tokens]
            traj = x[:, -trajectory_tokens:]
            traj = traj + self.cross_attn(traj, condition)
            x = torch.cat([prefix, traj], dim=1)

        # Apply the residual feed-forward update.
        mlp_raw = self.mlp(self.norm2(x))
        x = x + mlp_raw * self.mlp_scale
        return x

class PMFFinalLayer(nn.Module):
    """Project Pixel MeanFlow tokens into clean-data radar patches.

    The final linear layer is initialized to zero and is shared by the
    regular and forecast-horizon-dependent decoding paths.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))

    def forward_horizon(
        self,
        x: torch.Tensor,
        base_condition: torch.Tensor,
        horizon_condition: torch.Tensor,
        future_frames: int,
        channels_per_frame: int,
    ) -> torch.Tensor:
        """Decode each future frame using its horizon-dependent condition.

        The existing output projection weights are rearranged by forecast
        frame and channel; no additional learnable head is created.
        """
        if horizon_condition.ndim != 4:
            raise ValueError("horizon_condition must be [B,T,N,D].")
        b, t, n, d = horizon_condition.shape
        if t != int(future_frames):
            raise ValueError(
                f"horizon T={t} does not match future_frames={future_frames}."
            )
        if base_condition.shape != (b, n, d) or x.shape != (b, n, d):
            raise ValueError("x/base_condition shape mismatch for horizon decoding.")
        if self.out_channels != int(future_frames) * int(channels_per_frame):
            raise ValueError(
                "out_channels must equal future_frames * channels_per_frame."
            )

        # Add forecast-horizon-specific condition deviations before decoding.
        # The shared projection still produces the joint future sequence.
        delta = horizon_condition - base_condition.unsqueeze(1)
        x_h = self.norm(x.unsqueeze(1) + delta)  # [B,T,N,D]

        p = self.patch_size
        c = int(channels_per_frame)
        weight = self.linear.weight.view(p, p, t, c, d).permute(2, 0, 1, 3, 4)
        bias = self.linear.bias.view(p, p, t, c).permute(2, 0, 1, 3)

        # [B,T,N,D] x [T,P,P,C,D] -> [B,T,N,P,P,C]
        out = torch.einsum("btnd,tpqcd->btnpqc", x_h, weight)
        out = out + bias.view(1, t, 1, p, p, c)

        # Restore the final patch layout:
        # [B,N,P,P,T,C] -> [B,N,P*P*T*C].
        return out.permute(0, 2, 3, 4, 1, 5).contiguous().view(b, n, -1)
