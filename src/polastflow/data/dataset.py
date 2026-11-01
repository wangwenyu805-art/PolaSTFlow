from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class DualPolRadarDataset(Dataset):
    """Load normalized ZH/ZDR history-target sequence pairs from NPY files.

    Expected file layout::

        root/
          input/input_*.npy
          output/output_*.npy

    Each NPY is stored as [C,T,H,W]. Only the two dual-polarization channels
    ZH and ZDR are used. Disk-backed loading is intentionally kept simple;
    the previous RAM-cache configuration was removed because it duplicated
    configuration without changing the model itself.
    """

    def __init__(
        self,
        root: str,
        history_frames: int = 10,
        future_frames: int = 10,
        img_size: int = 256,
        radar_channels: int = 2,
        mmap_mode: str = "r",
        zh_min: float = 0.0,
        zh_max: float = 70.0,
        zdr_min: float = -1.0,
        zdr_max: float = 4.0,
    ) -> None:
        super().__init__()
        if radar_channels != 2:
            raise ValueError("DualPolRadarDataset expects exactly 2 channels: [ZH, ZDR].")

        self.root = Path(root)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        self.img_size = int(img_size)
        self.radar_channels = int(radar_channels)
        self.mmap_mode = mmap_mode
        self.zh_min = float(zh_min)
        self.zh_max = float(zh_max)
        self.zdr_min = float(zdr_min)
        self.zdr_max = float(zdr_max)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.input_dir}")
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Missing folder: {self.output_dir}")

        self.input_files: List[Path] = sorted(self.input_dir.glob("input_*.npy"))
        if not self.input_files:
            raise FileNotFoundError(f"No input_*.npy files found in {self.input_dir}")

        self.output_files = [
            self.output_dir / p.name.replace("input_", "output_", 1)
            for p in self.input_files
        ]
        missing = [str(p) for p in self.output_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing output files, first one: {missing[0]}")

        print(f"Dataset root: {self.root}")
        print(f"Samples: {len(self.input_files)}")
        print("Channels: ZH, ZDR")

    def __len__(self) -> int:
        return len(self.input_files)

    @staticmethod
    def _clip_normalize(x: np.ndarray, channel: int, vmin: float, vmax: float) -> None:
        x[channel] = np.clip(x[channel], vmin, vmax)
        x[channel] = (x[channel] - vmin) / (vmax - vmin)

    def _load_cthw(self, path: Path, expected_frames: int) -> np.ndarray:
        arr = np.load(path, mmap_mode=self.mmap_mode)
        arr = np.asarray(arr)

        expected = (
            self.radar_channels,
            expected_frames,
            self.img_size,
            self.img_size,
        )
        if arr.shape != expected:
            raise ValueError(f"{path} shape mismatch: got {arr.shape}, expected {expected}")

        arr = np.array(arr[: self.radar_channels], dtype=np.float32, copy=True)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_dualpol(self, x: np.ndarray) -> np.ndarray:
        self._clip_normalize(x, 0, self.zh_min, self.zh_max)
        self._clip_normalize(x, 1, self.zdr_min, self.zdr_max)
        return x

    def __getitem__(self, index: int):
        history = self._load_cthw(self.input_files[index], self.history_frames)
        target = self._load_cthw(self.output_files[index], self.future_frames)

        history = np.ascontiguousarray(self._normalize_dualpol(history), dtype=np.float32)
        target = np.ascontiguousarray(self._normalize_dualpol(target), dtype=np.float32)

        return (
            torch.from_numpy(history),
            torch.from_numpy(target),
            self.input_files[index].name,
        )


def collate_radar_sequences(batch):
    histories, targets, names = zip(*batch)
    history = torch.stack(histories, dim=0)
    target = torch.stack(targets, dim=0)

    if history.ndim != 5 or target.ndim != 5:
        raise ValueError("Batched radar tensors must be [B,C,T,H,W].")
    if history.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("Batched radar tensors must contain [ZH, ZDR].")

    return history, target, list(names)


class RadarPredictionDataset(Dataset):
    """Read input/input_*.npy only, with original dual-pol normalization."""
    def __init__(self, root: str, history_frames: int = 10, img_size: int = 256,
                 radar_channels: int = 2, mmap_mode: str = "r", zh_min: float = 0.0,
                 zh_max: float = 70.0, zdr_min: float = -1.0, zdr_max: float = 4.0):
        if radar_channels != 2:
            raise ValueError("RadarPredictionDataset requires [ZH, ZDR].")
        self.root = Path(root)
        self.input_dir = self.root / "input" if (self.root / "input").is_dir() else self.root
        self.input_files = sorted(self.input_dir.glob("input_*.npy"))
        if not self.input_files:
            raise FileNotFoundError(f"No input_*.npy files in {self.input_dir}")
        self.history_frames = history_frames
        self.img_size = img_size
        self.radar_channels = radar_channels
        self.mmap_mode = mmap_mode
        self.zh_min, self.zh_max = zh_min, zh_max
        self.zdr_min, self.zdr_max = zdr_min, zdr_max

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        path = self.input_files[index]
        x = np.asarray(np.load(path, mmap_mode=self.mmap_mode))
        shape = (2, self.history_frames, self.img_size, self.img_size)
        if x.shape != shape:
            raise ValueError(f"{path}: expected {shape}, received {x.shape}")
        x = np.array(x, dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        DualPolRadarDataset._clip_normalize(x, 0, self.zh_min, self.zh_max)
        DualPolRadarDataset._clip_normalize(x, 1, self.zdr_min, self.zdr_max)
        return torch.from_numpy(np.ascontiguousarray(x)), path.name


def collate_predictions(batch):
    histories, names = zip(*batch)
    return torch.stack(histories, dim=0), list(names)
