# PolaSTFlow

**PolaSTFlow: Physics-Aware One-Step Generation for Dual-Polarization Radar Echo Extrapolation**


## Overview

PolaSTFlow jointly predicts future horizontal reflectivity (ZH) and differential reflectivity (ZDR) sequences from historical radar observations. The framework incorporates physics-aware spatiotemporal modeling into one-step generation to support accurate and efficient radar echo extrapolation.

## Installation

Python 3.10 or later is required.

Install PyTorch according to your CUDA environment, then install the project dependencies:

```bash
pip install -e .
```

## Data Preparation

Organize the dataset as follows:

```text
dataset/
├── train/
│   ├── input/
│   └── output/
├── val/
│   ├── input/
│   └── output/
└── test/
    └── input/
```

Each `.npy` file contains two channels `[ZH, ZDR]` with shape `[2, 10, 256, 256]`, corresponding to `[C, T, H, W]`.

Training and validation require paired input and output files with matching sample identifiers.

## Training

**Single-GPU training:**

```bash
python scripts/train.py \
  --train-data dataset/train \
  --val-data dataset/val \
  --output-dir runs/default \
  --device cuda \
  --batch-size 4 \
  --epochs 100 \
  --lr 1e-4
```

**Multi-GPU training (DDP):**

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --train-data dataset/train \
  --val-data dataset/val \
  --output-dir runs/default \
  --device cuda \
  --batch-size 4 \
  --epochs 100 \
  --lr 1e-4
```

The training process saves two checkpoints:

- `best.pth`: Best checkpoint selected by validation loss.
- `last.pth`: Latest checkpoint for resuming training.

**Resume training:**

```bash
python scripts/train.py \
  --train-data dataset/train \
  --val-data dataset/val \
  --output-dir runs/default \
  --device cuda \
  --resume runs/default/last.pth
```

## Prediction

Generate future radar sequences using a trained checkpoint:

```bash
python scripts/predict.py \
  --data dataset/test \
  --checkpoint runs/default/best.pth \
  --output-dir predictions \
  --device cuda
```

Predictions are saved as `.npy` files. Ground-truth future frames are not required during inference.

## Citation

If you find PolaSTFlow useful in your research, please consider citing our paper.

The citation information will be updated upon publication.

## Acknowledgements

This work builds upon [Pixel MeanFlow (pMF)](https://github.com/Lyy-iiis/pMF). We thank the original authors for their contributions.
