from __future__ import annotations

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.layers import RMSNorm, SwiGLUFFN, get_2d_sincos_pos_embed
from ..common.patch import BottleneckPatchEmbedding


class PhysicsInformedObservationEncoder(nn.Module):
    """Encode PIKO observation features from historical ZH and ZDR fields.

    Eight channels describe first- and second-order temporal differences,
    first-order spatial variation of temporal changes and the discrete
    Laplacian of temporal changes for both polarimetric variables. Patch
    projection and refinement produce ``[B, T, N, D]`` observation tokens
    for DTLM's Kalman-guided correction.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        hidden_size: int,
        bottleneck_dim: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Temporal polarimetric evolution plus spatial derivatives of the
        # temporal-difference fields:
        # ΔZH, ΔZDR, Δ²ZH, Δ²ZDR,
        # |∇(ΔZH)|, |∇(ΔZDR)|, ∇²(ΔZH), ∇²(ΔZDR)
        physics_channels = 8

        self.patch_embed = BottleneckPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=physics_channels,
            pca_dim=bottleneck_dim,
            embed_dim=hidden_size,
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )

        self.norm = RMSNorm(hidden_size)

        self.refine = SwiGLUFFN(
            hidden_size,
            int(hidden_size * mlp_ratio),
            drop=proj_drop,
        )

        self.alpha_refine = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

        self.out_norm = RMSNorm(hidden_size)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        grid_size = int(self.patch_embed.num_patches ** 0.5)

        pos = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_size,
        )

        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0)
        )

    @staticmethod
    def _grad_abs(x: torch.Tensor) -> torch.Tensor:
        """Approximate absolute first-order spatial variation for each channel."""

        dx = torch.abs(
            x[:, :, :, 1:] - x[:, :, :, :-1]
        )

        dy = torch.abs(
            x[:, :, 1:, :] - x[:, :, :-1, :]
        )

        dx = F.pad(
            dx,
            (0, 1, 0, 0),
            mode="replicate",
        )

        dy = F.pad(
            dy,
            (0, 0, 0, 1),
            mode="replicate",
        )

        return dx + dy

    @staticmethod
    def _laplacian(x: torch.Tensor) -> torch.Tensor:
        """Compute a discrete spatial Laplacian independently per channel."""

        x_pad = F.pad(
            x,
            (1, 1, 1, 1),
            mode="replicate",
        )

        center = x_pad[:, :, 1:-1, 1:-1]

        left = x_pad[:, :, 1:-1, :-2]
        right = x_pad[:, :, 1:-1, 2:]
        up = x_pad[:, :, :-2, 1:-1]
        down = x_pad[:, :, 2:, 1:-1]

        laplacian = (
            left
            + right
            + up
            + down
            - 4.0 * center
        )

        return laplacian

    def _build_physics_maps(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:

        if history.ndim != 5:
            raise ValueError(
                f"history must be [B,C,T,H,W], "
                f"got {tuple(history.shape)}"
            )

        b, c, t, h, w = history.shape

        if c < 2:
            raise ValueError(
                "PhysicsInformedObservationEncoder requires "
                "ZH and ZDR channels."
            )

        # --------------------------------------------------
        # Original radar variables
        #
        # They are used only to construct temporal and
        # spatial differential descriptors.
        # They are not directly included in physics maps.
        # --------------------------------------------------

        zh_seq = history[:, 0:1]
        zdr_seq = history[:, 1:2]

        # --------------------------------------------------
        # 1. First-order temporal evolution
        #
        # ΔZH_t  = ZH_t  - ZH_(t-1)
        # ΔZDR_t = ZDR_t - ZDR_(t-1)
        #
        # The first frame is set to zero.
        # --------------------------------------------------

        dzh_seq = torch.zeros_like(zh_seq)
        dzdr_seq = torch.zeros_like(zdr_seq)

        if t > 1:
            dzh_seq[:, :, 1:] = (
                zh_seq[:, :, 1:]
                - zh_seq[:, :, :-1]
            )

            dzdr_seq[:, :, 1:] = (
                zdr_seq[:, :, 1:]
                - zdr_seq[:, :, :-1]
            )

        # --------------------------------------------------
        # 2. Second-order temporal evolution
        #
        # Δ²ZH_t =
        #     ZH_t - 2*ZH_(t-1) + ZH_(t-2)
        #
        # Δ²ZDR_t =
        #     ZDR_t - 2*ZDR_(t-1) + ZDR_(t-2)
        #
        # The first two frames are set to zero.
        # --------------------------------------------------

        ddzh_seq = torch.zeros_like(zh_seq)
        ddzdr_seq = torch.zeros_like(zdr_seq)

        if t > 2:
            ddzh_seq[:, :, 2:] = (
                zh_seq[:, :, 2:]
                - 2.0 * zh_seq[:, :, 1:-1]
                + zh_seq[:, :, :-2]
            )

            ddzdr_seq[:, :, 2:] = (
                zdr_seq[:, :, 2:]
                - 2.0 * zdr_seq[:, :, 1:-1]
                + zdr_seq[:, :, :-2]
            )

        # [B,C,T,H,W]
        #       ↓
        # [B*T,C,H,W]

        dzh = (
            dzh_seq
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, 1, h, w)
        )

        dzdr = (
            dzdr_seq
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, 1, h, w)
        )

        ddzh = (
            ddzh_seq
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, 1, h, w)
        )

        ddzdr = (
            ddzdr_seq
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, 1, h, w)
        )

        # --------------------------------------------------
        # 3. Spatial gradient of first-order temporal differences
        #
        # These gradients describe spatial variation in radar echo evolution;
        # PSR separately represents the spectral structure of the input fields.
        # --------------------------------------------------

        grad_dzh = self._grad_abs(dzh)
        grad_dzdr = self._grad_abs(dzdr)

        # --------------------------------------------------
        # 4. Second-order spatial gradient of first-order temporal differences
        #
        # The discrete Laplacian preserves the sign of local curvature and
        # characterizes compact generation/decay structures in ΔZH/ΔZDR.
        # --------------------------------------------------

        lap_dzh = self._laplacian(dzh)
        lap_dzdr = self._laplacian(dzdr)

        # --------------------------------------------------
        # Physics feature maps
        #
        # 8 channels:
        # 0: ΔZH
        # 1: ΔZDR
        # 2: Δ²ZH
        # 3: Δ²ZDR
        # 4: |∇(ΔZH)|
        # 5: |∇(ΔZDR)|
        # 6: ∇²(ΔZH)
        # 7: ∇²(ΔZDR)
        # --------------------------------------------------

        return torch.cat(
            [
                dzh,
                dzdr,
                ddzh,
                ddzdr,
                grad_dzh,
                grad_dzdr,
                lap_dzh,
                lap_dzdr,
            ],
            dim=1,
        )

    def forward_sequence(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:

        b, _, t, _, _ = history.shape

        # Construct physics descriptors
        # [B*T, 8, H, W]
        physics = self._build_physics_maps(history)

        # Patch embedding
        # [B*T, 8, H, W]
        #       ↓
        # [B*T, N, D]
        z = self.patch_embed(physics)

        # Spatial positional encoding
        z = (
            z
            + self.pos_embed.to(
                dtype=z.dtype,
                device=z.device,
            )
        )

        # Nonlinear refinement
        z = (
            z
            + self.alpha_refine
            * self.refine(self.norm(z))
        )

        z = self.out_norm(z)

        # [B*T,N,D]
        #      ↓
        # [B,T,N,D]
        n, d = z.shape[1], z.shape[2]

        return z.view(b, t, n, d)
