from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import jvp

from .backbone import PixelMeanFlowSequenceBackbone


class PolaSTFlow(nn.Module):
    """Physics-aware one-step generation of future dual-polarization radar echoes.

    The historical condition encoder guides the Pixel MeanFlow backbone.
    Training retains the original interval sampling, auxiliary velocity
    branch, MeanFlow JVP objective and residual losses. Inference maps one
    Gaussian noise sample into the complete future ZH/ZDR sequence.
    """

    def __init__(
        self,
        backbone: PixelMeanFlowSequenceBackbone,
        time_logit_mean: float = -0.4,
        time_logit_std: float = 1.0,
        noise_scale: float = 1.0,
        data_proportion: float = 0.5,
        velocity_eps: float = 5e-2,
        norm_p: float = 0.0,
        norm_eps: float = 1e-2,
        aux_v_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.time_logit_mean = float(time_logit_mean)
        self.time_logit_std = float(time_logit_std)
        self.noise_scale = float(noise_scale)
        self.data_proportion = float(data_proportion)
        self.velocity_eps = float(velocity_eps)
        self.norm_p = float(norm_p)
        self.norm_eps = float(norm_eps)
        self.aux_v_weight = float(aux_v_weight)

    @staticmethod
    def _broadcast(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, *([1] * (x.ndim - 1)))

    def _sample_time(self, n: int, device, dtype) -> torch.Tensor:
        z = torch.randn(n, device=device, dtype=torch.float32)
        return torch.sigmoid(z * self.time_logit_std + self.time_logit_mean).to(dtype)

    def _sample_interval(self, n: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._sample_time(n, device, dtype)
        r = self._sample_time(n, device, dtype)
        r = torch.where(torch.rand(n, device=device) < self.data_proportion, t, r)
        return torch.maximum(t, r), torch.minimum(t, r)

    def _velocity_from_clean(
        self, z_t: torch.Tensor, clean: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (z_t - clean) / self._broadcast(t, z_t).clamp_min(self.velocity_eps)

    def _loss(self, diff: torch.Tensor) -> torch.Tensor:
        value = diff.float().square().flatten(1).sum(dim=1)
        if self.norm_p != 0.0:
            weight = (value.detach() + self.norm_eps).pow(self.norm_p)
            value = value / weight.clamp_min(1e-12)
        return value.mean()

    def forward(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_logs: bool = False,
    ):
        if history.ndim != 5 or target.ndim != 5:
            raise ValueError("history and target must be [B,C,T,H,W].")

        b = target.shape[0]
        if t is None:
            t, r = self._sample_interval(b, target.device, target.dtype)
        else:
            t = t.to(target).reshape(-1)
            r = torch.zeros_like(t) if r is None else r.to(target).reshape(-1)
            t, r = torch.maximum(t, r), torch.minimum(t, r)

        if noise is None:
            noise = torch.randn_like(target) * self.noise_scale
        else:
            noise = noise.to(target) * self.noise_scale

        z_t = (1.0 - self._broadcast(t, target)) * target + self._broadcast(t, target) * noise
        target_v = self._velocity_from_clean(z_t, target, t)
        cond = self.backbone.encode_history(history)

        # The instantaneous-velocity branch supplies the tangent for the
        # MeanFlow Jacobian-vector product (JVP).
        zeros = torch.zeros_like(t)
        _, clean_v0 = self.backbone.forward_with_condition(
            z_t, zeros, cond, need_u=False, need_v=True
        )
        tangent_v = self._velocity_from_clean(z_t, clean_v0, t).detach()

        def u_fn(z_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor):
            interval = (t_in - r_in).clamp_min(0.0)
            clean_u, clean_v = self.backbone.forward_with_condition(
                z_in, interval, cond, need_u=True, need_v=True
            )
            u = self._velocity_from_clean(z_in, clean_u, t_in)
            v = self._velocity_from_clean(z_in, clean_v, t_in)
            return u, (v, clean_u)

        u_raw, du_dt_raw, aux = jvp(
            u_fn,
            (z_t, t, r),
            (tangent_v, torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
        v_aux_raw, clean_u = aux

        meanflow_raw = u_raw + self._broadcast(t - r, z_t) * du_dt_raw.detach()

        diff_mf = meanflow_raw - target_v.detach()
        diff_v = v_aux_raw - target_v.detach()
        loss_mf = self._loss(diff_mf)
        loss_v = self._loss(diff_v)
        raw_mf = diff_mf.float().square().mean()
        raw_v = diff_v.float().square().mean()
        raw_total = raw_mf + self.aux_v_weight * raw_v
        loss = loss_mf + self.aux_v_weight * loss_v

        if not return_logs:
            return loss

        logs: Dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "objective_mf": loss_mf.detach(),
            "objective_v": loss_v.detach(),
            "meanflow": raw_mf.detach(),
            "aux_v": raw_v.detach(),
            "raw_total": raw_total.detach(),
            "clean_mae": (clean_u.detach() - target).abs().mean(),
        }
        return loss, logs

    @torch.no_grad()
    def generate(
        self,
        history: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if history.ndim != 5:
            raise ValueError("history must be [B,C,T,H,W].")

        b, _, _, h, w = history.shape
        shape = (b, self.backbone.radar_channels, self.backbone.future_frames, h, w)
        z_1 = torch.randn(
            shape,
            device=history.device,
            dtype=history.dtype,
            generator=generator,
        ) * self.noise_scale

        cond = self.backbone.encode_history(history)
        t = torch.ones(b, device=history.device, dtype=history.dtype)
        r = torch.zeros_like(t)
        interval = t - r
        clean, _ = self.backbone.forward_with_condition(
            z_1, interval, cond, need_u=True, need_v=False
        )
        raw_u = self._velocity_from_clean(z_1, clean, t)
        return z_1 - self._broadcast(interval, z_1) * raw_u
