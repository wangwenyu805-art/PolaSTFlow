from __future__ import annotations
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm
from .spectral import FixedSpectralDecomposition
from .blocks import PatchContextBlock, ModeRefiner

class PolarimetricSpectralRepresentation(nn.Module):
    """Construct polarimetric spectral features from ZH and ZDR frames.

    PSR decomposes both radar variables into low, middle and high spatial
    frequency bands. Each band has separate ZH/ZDR extractors followed by
    common and differential polarimetric mode refinement. Their projected
    responses form fused tokens and explicit low/mid/high band tokens,
    from which the historical condition encoder derives fast/slow inputs.
    """

    BAND_NAMES = ("low", "mid", "high")

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        hidden_size: int,
        band_dim: int | None = None,
        proj_drop: float = 0.0,
        sigma_small: float = 1.0,
        sigma_large: float = 2.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.band_dim = int(band_dim or max(32, hidden_size // 6))

        self.spectral = FixedSpectralDecomposition(
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            kernel_small=5,
            kernel_large=9,
        )

        # Frequency-dependent contexts: wider low-band ZH context and local ZDR detail.
        # These receptive fields are fixed rather than dynamically gated.
        zh_kernels = {"low": 5, "mid": 3, "high": 3}
        zdr_kernels = {"low": 3, "mid": 3, "high": 3}

        self.zh_extractors = nn.ModuleDict({
            name: PatchContextBlock(
                1,
                self.band_dim,
                patch_size=self.patch_size,
                context_kernel=zh_kernels[name],
                drop=proj_drop,
            )
            for name in self.BAND_NAMES
        })
        self.zdr_extractors = nn.ModuleDict({
            name: PatchContextBlock(
                1,
                self.band_dim,
                patch_size=self.patch_size,
                context_kernel=zdr_kernels[name],
                drop=proj_drop,
            )
            for name in self.BAND_NAMES
        })

        # Refine common and differential polarimetric modes separately.
        # The low-frequency common mode uses the widest refinement kernel.
        self.common_refiners = nn.ModuleDict({
            "low": ModeRefiner(self.band_dim, 5, proj_drop),
            "mid": ModeRefiner(self.band_dim, 3, proj_drop),
            "high": ModeRefiner(self.band_dim, 3, proj_drop),
        })
        self.diff_refiners = nn.ModuleDict({
            name: ModeRefiner(self.band_dim, 3, proj_drop)
            for name in self.BAND_NAMES
        })

        self.mode_projectors = nn.ModuleDict()
        for mode in ("common", "diff"):
            for band in self.BAND_NAMES:
                self.mode_projectors[f"{mode}_{band}"] = nn.Conv2d(
                    self.band_dim,
                    hidden_size,
                    kernel_size=1,
                    bias=False,
                )

        self.output_refine = nn.Sequential(
            nn.GroupNorm(1, hidden_size),
            nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                padding=1,
                groups=hidden_size,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
        )
        self.out_norm = RMSNorm(hidden_size)
        self.last_stats: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _energy(x: torch.Tensor) -> torch.Tensor:
        return x.float().square().mean()

    def get_diagnostics(self) -> Dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in self.last_stats.items()
        }

    def _encode_frame(
        self,
        frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one dual-polarization frame into fused and band-wise PSR tokens.

        Returns fused tokens of shape ``[B, N, D]`` and band tokens of shape
        ``[B, 3, N, D]``, ordered as low, middle and high frequency.
        """
        if frame.ndim != 4:
            raise ValueError(f"frame must be [B,C,H,W], got {tuple(frame.shape)}")
        if frame.shape[1] != 2:
            raise ValueError(
                "PSP encoder expects exactly two channels ordered as [ZH,ZDR], "
                f"got C={frame.shape[1]}."
            )
        if frame.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected spatial size {(self.img_size, self.img_size)}, "
                f"got {tuple(frame.shape[-2:])}."
            )

        zh = frame[:, 0:1]
        zdr = frame[:, 1:2]
        zh_bands = dict(zip(self.BAND_NAMES, self.spectral(zh)))
        zdr_bands = dict(zip(self.BAND_NAMES, self.spectral(zdr)))

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        band_outputs = []
        common_energy = []
        diff_energy = []

        for band in self.BAND_NAMES:
            f_zh = self.zh_extractors[band](zh_bands[band])
            f_zdr = self.zdr_extractors[band](zdr_bands[band])

            # Form orthogonal common and differential ZH/ZDR features.
            common = (f_zh + f_zdr) * inv_sqrt2
            diff = (f_zh - f_zdr) * inv_sqrt2

            common = self.common_refiners[band](common)
            diff = self.diff_refiners[band](diff)

            common_out = self.mode_projectors[f"common_{band}"](common)
            diff_out = self.mode_projectors[f"diff_{band}"](diff)

            # Normalize each band-wise common/differential sum by sqrt(2).
            band_outputs.append((common_out + diff_out) * inv_sqrt2)
            common_energy.append(self._energy(common))
            diff_energy.append(self._energy(diff))

        bands = torch.stack(band_outputs, dim=1)  # [B,3,D,G,G]

        # Aggregate the three normalized frequency-band outputs into fused PSR features.
        fused = bands.sum(dim=1) / math.sqrt(len(self.BAND_NAMES))
        fused = fused + self.output_refine(fused)

        b, d, gh, gw = fused.shape
        if gh != self.grid_size or gw != self.grid_size:
            raise RuntimeError(
                f"Unexpected PSP grid {(gh, gw)}, expected "
                f"{(self.grid_size, self.grid_size)}."
            )

        fused_tokens = fused.flatten(2).transpose(1, 2).contiguous()
        fused_tokens = self.out_norm(fused_tokens)

        # Apply the shared output normalization to fused and per-band tokens.
        band_tokens = bands.flatten(3).permute(0, 1, 3, 2).contiguous()
        band_tokens = self.out_norm(band_tokens)

        with torch.no_grad():
            eps = 1e-12
            for prefix, spectral_bands in (("zh", zh_bands), ("zdr", zdr_bands)):
                energies = torch.stack([
                    self._energy(spectral_bands[name]) for name in self.BAND_NAMES
                ])
                ratios = energies / energies.sum().clamp_min(eps)
                for idx, band in enumerate(self.BAND_NAMES):
                    self.last_stats[f"psp_{prefix}_{band}_energy"] = ratios[idx]
            c_energy = torch.stack(common_energy).mean()
            p_energy = torch.stack(diff_energy).mean()
            denom = (c_energy + p_energy).clamp_min(eps)
            self.last_stats["psp_common_mode_energy"] = c_energy / denom
            self.last_stats["psp_diff_mode_energy"] = p_energy / denom

        return fused_tokens, band_tokens

    def forward_with_bands(
        self,
        frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fused PSR tokens together with low/mid/high band tokens."""
        return self._encode_frame(frame)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # The standard forward path returns the fused PSR token stream.
        fused_tokens, _ = self._encode_frame(frame)
        return fused_tokens
