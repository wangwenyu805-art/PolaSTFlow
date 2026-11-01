# PolaSTFlow

PyTorch implementation of **PolaSTFlow: Physics-Aware One-Step Generation for Dual-Polarization Radar Echo Extrapolation**.

The manuscript is currently under review at *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*. This repository provides the implementation for peer-review purposes.

## Requirements

Python 3.10 or later is required.

Install PyTorch according to your CUDA environment, then install the project dependencies:

```bash
pip install -e .
```

## Data Preparation

The model uses historical ZH and ZDR radar sequences to predict future radar observations.

Each `.npy` file has the shape `[2, 10, 256, 256]`, corresponding to `[C, T, H, W]`.

Organize the data as follows:

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

Training and validation require paired input and output files with matching sample identifiers. The dataset is not included in this repository.

## Training

Single-GPU training:

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

Multi-GPU training:

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

## Prediction

```bash
python scripts/predict.py \
  --data dataset/test \
  --checkpoint runs/default/best.pth \
  --output-dir predictions \
  --device cuda
```

## Acknowledgements

This implementation builds upon [Pixel MeanFlow (pMF)](https://github.com/Lyy-iiis/pMF).
