from __future__ import annotations
import csv
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from polastflow.data import DualPolRadarDataset, collate_radar_sequences

TRAIN_SEED = 42


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def require_bcthw(x: torch.Tensor, name: str):
    if x.ndim != 5:
        raise ValueError(f'{name} must be [B,C,T,H,W], got {tuple(x.shape)}')
    return x


def seed_training(rank: int):
    """Called once per process, not once per epoch or batch."""
    seed = TRAIN_SEED + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main_process():
    return get_rank() == 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def setup_device(args):
    """Preserve one process per GPU to avoid JVP / DataParallel incompatibility."""
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP training requires CUDA/NCCL (same as the original code).')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        args.distributed = True
        args.local_rank = local_rank
        args.rank = int(os.environ['RANK'])
        args.world_size = dist.get_world_size()
        return torch.device(f'cuda:{local_rank}')
    args.distributed = False
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1
    name = str(args.device).lower()
    if name.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA requested but unavailable: {args.device}')
        device = torch.device(name if ':' in name else 'cuda:0')
        torch.cuda.set_device(device)
        return device
    if name == 'cpu':
        return torch.device('cpu')
    raise ValueError('device must be cpu or cuda[:index]')


def wrap_model_for_multi_gpu(model, args):
    if args.distributed:
        return DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                   broadcast_buffers=False, find_unused_parameters=True)
    return model


def make_loader(root, cfg, batch_size, workers, shuffle, drop_last, sampler=None):
    ds = DualPolRadarDataset(root=root, history_frames=cfg.history_frames,
         future_frames=cfg.future_frames, img_size=cfg.img_size,
         radar_channels=cfg.radar_channels, zh_min=cfg.zh_min, zh_max=cfg.zh_max,
         zdr_min=cfg.zdr_min, zdr_max=cfg.zdr_max)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
         sampler=sampler, num_workers=workers, pin_memory=True, drop_last=drop_last,
         collate_fn=collate_radar_sequences)


def build_train_val_loaders(args, cfg):
    if args.distributed:
        if args.batch_size < args.world_size or args.batch_size % args.world_size:
            raise ValueError('global batch-size must be divisible by world size')
        local_batch = args.batch_size // args.world_size
        train_ds = DualPolRadarDataset(root=args.train_data, history_frames=cfg.history_frames,
             future_frames=cfg.future_frames, img_size=cfg.img_size,
             radar_channels=cfg.radar_channels, zh_min=cfg.zh_min, zh_max=cfg.zh_max,
             zdr_min=cfg.zdr_min, zdr_max=cfg.zdr_max)
        sampler = DistributedSampler(train_ds, num_replicas=args.world_size,
                                    rank=args.rank, shuffle=True, seed=TRAIN_SEED, drop_last=True)
        train_loader = DataLoader(train_ds, batch_size=local_batch, shuffle=False,
             sampler=sampler, num_workers=args.num_workers, pin_memory=True,
             drop_last=True, collate_fn=collate_radar_sequences)
    else:
        train_loader = make_loader(args.train_data, cfg, args.batch_size, args.num_workers,
                                   shuffle=True, drop_last=True)
    val_loader = None
    if is_main_process():
        val_loader = make_loader(args.val_data, cfg, args.val_batch_size, args.num_workers,
                                 shuffle=False, drop_last=False)
    return train_loader, val_loader


def append_csv(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
