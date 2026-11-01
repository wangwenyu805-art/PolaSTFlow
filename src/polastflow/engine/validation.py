from __future__ import annotations
import torch
from tqdm import tqdm
from .utils import require_bcthw, unwrap_model

VALIDATION_SEED = 12345


def validation_loss(model, data_loader, device, amp: bool = False):
    """Unchanged original validation raw_total with fixed per-run noise sequence."""
    model.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(VALIDATION_SEED)
    core = unwrap_model(model)
    with torch.no_grad():
        for history, target, _ in tqdm(data_loader, desc='Validation loss', leave=False):
            history = require_bcthw(history.to(device).float(), 'history')
            target = require_bcthw(target.to(device).float(), 'target')
            b = target.shape[0]
            z_t = torch.randn(b, generator=generator, device=device)
            z_r = torch.randn(b, generator=generator, device=device)
            t = torch.sigmoid(z_t * core.time_logit_std + core.time_logit_mean)
            r = torch.sigmoid(z_r * core.time_logit_std + core.time_logit_mean)
            fm_mask = torch.rand(b, generator=generator, device=device) < core.data_proportion
            r = torch.where(fm_mask, t, r)
            t, r = torch.maximum(t, r), torch.minimum(t, r)
            noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                _, logs = model(history, target, t=t, r=r, noise=noise, return_logs=True)
            raw_total = logs['raw_total']
            if torch.is_tensor(raw_total) and raw_total.ndim > 0:
                raw_total = raw_total.mean()
            total += float(raw_total.detach().cpu()) * b
            count += b
    return 0.0 if not count else total / count
