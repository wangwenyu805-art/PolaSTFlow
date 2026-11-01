from __future__ import annotations
import math
import os
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from polastflow.models.factory import build_model
from polastflow.config import load_model_config
from .checkpoint import save_checkpoint, load_training_checkpoint
from .utils import (seed_training, setup_device, wrap_model_for_multi_gpu,
                    build_train_val_loaders, append_csv, is_main_process,
                    is_distributed, get_rank, unwrap_model, require_bcthw, cleanup_distributed)
from .validation import validation_loss


def train(args):
    device = setup_device(args)
    seed_training(get_rank())
    cfg = load_model_config(args.model_config)
    main = is_main_process()
    if main: os.makedirs(args.output_dir, exist_ok=True)
    train_loader, val_loader = build_train_val_loaders(args, cfg)
    model = wrap_model_for_multi_gpu(build_model(cfg).to(device), args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                 weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp))
    start_epoch, best_val_loss = 0, float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_training_checkpoint(
            args.resume, model, optimizer, scaler, device,
            restore_rng_state=(not args.distributed or main))
        if main: print(f'Resume from epoch {start_epoch+1}, best val={best_val_loss:.8f}')
    if main:
        print(f'PolaSTFlow parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}')
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    try:
        for epoch in range(start_epoch, args.epochs):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            model.train()
            epoch_start = time.time()
            running_loss, batches = 0.0, 0
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=True,
                            disable=not main)
            for history, target, _ in iterator:
                history = require_bcthw(history.to(device, non_blocking=True).float(), 'history')
                target = require_bcthw(target.to(device, non_blocking=True).float(), 'target')
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type=='cuda'):
                    loss, logs = model(history, target, return_logs=True)
                if loss.ndim > 0:
                    loss = loss.mean()
                logs = {k: v.mean() if torch.is_tensor(v) and v.ndim > 0 else v
                        for k, v in logs.items()}
                if not torch.isfinite(loss).item():
                    if main: print('[Warning] non-finite loss; batch skipped.')
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_value = float(grad_norm.detach().cpu())
                if not math.isfinite(grad_value):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    if main: print('[Warning] non-finite gradient; batch skipped.')
                    continue
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(logs['raw_total'].detach().cpu())
                batches += 1
                if main:
                    iterator.set_postfix(raw=f'{running_loss/batches:.5f}',
                        objective=f'{float(loss.detach().cpu()):.3f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
            if is_distributed():
                sums = torch.tensor([running_loss, float(batches)],
                                  dtype=torch.float64, device=device)
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                running_loss, batches = float(sums[0].item()), int(sums[1].item())
            train_raw_total = running_loss / max(batches, 1)
            if is_distributed(): dist.barrier()
            if main:
                val_raw_total = validation_loss(unwrap_model(model), val_loader, device, amp=args.amp)
                improved = val_raw_total < best_val_loss
                if improved:
                    best_val_loss = val_raw_total
                    save_checkpoint(os.path.join(args.output_dir, 'best.pth'), model,
                                    optimizer, scaler, epoch, best_val_loss, cfg, args)
                save_checkpoint(os.path.join(args.output_dir, 'last.pth'), model,
                                optimizer, scaler, epoch, best_val_loss, cfg, args)
                append_csv(log_path, dict(epoch=epoch+1, train_raw_total=train_raw_total,
                                         val_raw_total=val_raw_total))
                print(f'Epoch {epoch+1}: train_raw_total={train_raw_total:.8f}, '
                      f'val_raw_total={val_raw_total:.8f}, best={best_val_loss:.8f}, '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}, '
                      f'best_saved={improved}, duration={time.time()-epoch_start:.1f}s')
            if is_distributed(): dist.barrier()
    finally:
        cleanup_distributed()
