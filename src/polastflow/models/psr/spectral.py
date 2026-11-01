from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Construct a normalized two-dimensional Gaussian convolution kernel."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords.square()) / (2.0 * float(sigma) ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()

class FixedSpectralDecomposition(nn.Module):
    """Split a radar field into fixed low-, middle- and high-frequency bands.

    Two Gaussian smoothers give ``low = G_large(X)``,
    ``mid = G_small(X) - G_large(X)``, and
    ``high = X - G_small(X)``. Their sum reconstructs the input, apart
    from floating-point rounding.
    """

    def __init__(
        self,
        sigma_small: float = 1.0,
        sigma_large: float = 2.0,
        kernel_small: int = 5,
        kernel_large: int = 9,
    ) -> None:
        super().__init__()
        if not (0.0 < sigma_small < sigma_large):
            raise ValueError("Expected 0 < sigma_small < sigma_large.")
        self.kernel_small = int(kernel_small)
        self.kernel_large = int(kernel_large)
        self.register_buffer(
            "gaussian_small",
            _gaussian_kernel2d(self.kernel_small, sigma_small)[None, None],
            persistent=False,
        )
        self.register_buffer(
            "gaussian_large",
            _gaussian_kernel2d(self.kernel_large, sigma_large)[None, None],
            persistent=False,
        )

    @staticmethod
    def _blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = kernel.shape[-1] // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, kernel.to(device=x.device, dtype=x.dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
        smooth_small = self._blur(x, self.gaussian_small)
        low = self._blur(x, self.gaussian_large)
        mid = smooth_small - low
        high = x - smooth_small
        return low, mid, high
