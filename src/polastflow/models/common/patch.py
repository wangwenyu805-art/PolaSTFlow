import torch
import torch.nn as nn


class BottleneckPatchEmbedding(nn.Module):
    """Embed non-overlapping image patches into the model token space.

    Flatten each patch using ``nn.Unfold`` and project it through the
    existing low-dimensional bottleneck into the Transformer hidden size.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        pca_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj1 = nn.Linear(patch_dim, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"input size {tuple(x.shape[-2:])} != {(self.img_size, self.img_size)}"
            )
        x = self.unfold(x).transpose(1, 2).contiguous()
        return self.proj2(self.proj1(x))
