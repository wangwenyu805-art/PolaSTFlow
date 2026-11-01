from __future__ import annotations
import os
import random
import numpy as np
import torch
from .utils import unwrap_model


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    if not state: return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss, model_config, train_args):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'model': unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': int(epoch), 'best_val_loss': float(best_val_loss),
                'model_config': dict(vars(model_config)),
                'train_args': dict(vars(train_args)),
                'rng_state': rng_state()}, path)


def load_model_state(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    unwrap_model(model).load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


def load_training_checkpoint(path, model, optimizer, scaler, device, restore_rng_state=True):
    ckpt = load_model_state(path, model, device)
    if ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(device)
    if ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
    if restore_rng_state:
        restore_rng(ckpt.get('rng_state'))
    val_best = ckpt.get('best_val_loss', ckpt.get('metrics', {}).get('best_val_loss', float('inf')))
    return int(ckpt.get('epoch', -1))+1, float(val_best)
