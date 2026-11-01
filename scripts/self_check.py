#!/usr/bin/env python3
"""Lightweight forward/backward/sampling check, independent of an installed dataset."""
import torch
from types import SimpleNamespace
from pathlib import Path
from polastflow.config import load_model_config
from polastflow.models.factory import build_model


def main():
    cfg = vars(load_model_config(Path(__file__).resolve().parents[1] / 'configs/model.yaml'))
    cfg.update(img_size=32, history_frames=2, future_frames=2, hidden_size=64,
               prediction_depth=3, pmf_aux_head_depth=1, num_heads=4,
               bottleneck_dim=32, condition_bottleneck_dim=24,
               pmf_interval_tokens=2)
    torch.manual_seed(42)
    model = build_model(SimpleNamespace(**cfg))
    history = torch.rand(1, 2, 2, 32, 32)
    target = torch.rand_like(history)
    loss, logs = model(history, target, return_logs=True)
    loss.backward()
    model.eval()
    with torch.no_grad():
        generated = model.generate(history, generator=torch.Generator().manual_seed(12345))
    assert generated.shape == target.shape
    print(f'Self-check passed; objective={loss.item():.6f}; raw_total={logs["raw_total"].item():.6f}')


if __name__ == '__main__':
    main()
