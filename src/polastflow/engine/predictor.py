from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from polastflow.config import load_model_config
from polastflow.models.factory import build_model
from polastflow.data import RadarPredictionDataset, collate_predictions
from .checkpoint import load_model_state

SAMPLING_SEED = 12345


def predict(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    cfg = load_model_config(args.model_config)
    ds = RadarPredictionDataset(args.data, cfg.history_frames, cfg.img_size,
        cfg.radar_channels, zh_min=cfg.zh_min, zh_max=cfg.zh_max,
        zdr_min=cfg.zdr_min, zdr_max=cfg.zdr_max)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
         num_workers=args.num_workers, pin_memory=True,
         collate_fn=collate_predictions)
    model = build_model(cfg).to(device)
    load_model_state(args.checkpoint, model, device)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(SAMPLING_SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for history, names in tqdm(loader, desc='Predict'):
            history = history.to(device).float()
            with torch.amp.autocast(device_type=device.type,
                                    enabled=args.amp and device.type == 'cuda'):
                pred = model.generate(history, generator=generator)
            arr = pred.detach().float().cpu().numpy()
            for index, name in enumerate(names):
                safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', Path(name).stem)
                path = Path(args.output_dir) / f'{count:06d}_{safe}_pred.npy'
                np.save(path, arr[index])
                count += 1
    print(f'Saved {count} predictions to {args.output_dir}')
