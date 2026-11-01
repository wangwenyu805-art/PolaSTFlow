from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common.layers import RMSNorm, SwiGLUFFN

class KalmanGuidedLiquidCell(nn.Module):
    """Update fast/slow liquid states and apply physics-informed Kalman correction.

    PSR-conditioned tokens drive the recurrent state transitions and the
    adaptive fast/slow time constants. PIKO tokens provide observation
    increments for separate fast and slow Kalman updates. The uncorrected
    recurrent memory is carried to the next historical frame, while the
    corrected states are exposed as historical conditions. No geometric
    feature warping or motion alignment is performed.
    """

    def __init__(self, dim: int, min_tau: float=0.1, fast_tau_max: float=2.0, max_tau: float=10.0, motion_kernel_size: int=3, eps: float=1e-06) -> None:
        super().__init__()
        if not 0.0 < min_tau < max_tau:
            raise ValueError(f'Expected 0 < min_tau < max_tau, got {min_tau}, {max_tau}.')
        if not min_tau < fast_tau_max < max_tau:
            raise ValueError(f'For dual-timescale mode expected min_tau < fast_tau_max < max_tau, got {min_tau}, {fast_tau_max}, {max_tau}.')
        self.dim = int(dim)
        self.min_tau = float(min_tau)
        self.fast_tau_max = float(fast_tau_max)
        self.max_tau = float(max_tau)
        self.eps = float(eps)
        self.motion_kernel_size = int(motion_kernel_size)
        self.temporal_fusion = nn.Sequential(RMSNorm(2 * dim), nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.parameter_norm = RMSNorm(dim)
        self.core_param_dim = 6 * dim
        self.state_count = 2
        self.parameter_generator = nn.Linear(dim, self.core_param_dim)
        self.physics_delta_proj = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, self.state_count * dim))
        self.kalman_control_norm = RMSNorm(2 * dim)
        self.kalman_parameter_generator = nn.Linear(2 * dim, 3 * self.state_count)
        self.direct_physics_gate = nn.Linear(2 * dim, self.state_count)

    def reset_special_initialization(self) -> None:
        """Initialize the liquid-memory and Kalman parameters for stable training."""
        nn.init.zeros_(self.parameter_generator.bias)
        last_fusion = self.temporal_fusion[-1]
        nn.init.normal_(last_fusion.weight, mean=0.0, std=0.001)
        nn.init.zeros_(last_fusion.bias)
        last_phys = self.physics_delta_proj[-1]
        nn.init.normal_(last_phys.weight, mean=0.0, std=0.001)
        nn.init.zeros_(last_phys.bias)
        nn.init.zeros_(self.kalman_parameter_generator.weight)
        nn.init.zeros_(self.kalman_parameter_generator.bias)
        with torch.no_grad():
            self.kalman_parameter_generator.bias[:self.state_count].fill_(2.0)
            self.kalman_parameter_generator.bias[self.state_count:2 * self.state_count].fill_(-2.0)
            self.kalman_parameter_generator.bias[2 * self.state_count:3 * self.state_count].zero_()
        nn.init.zeros_(self.direct_physics_gate.weight)
        nn.init.constant_(self.direct_physics_gate.bias, -2.0)

    def _dual_time_constants(self, raw_tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_fast, raw_gap = raw_tau.chunk(2, dim=-1)
        tau_fast = self.min_tau + (self.fast_tau_max - self.min_tau) * torch.sigmoid(raw_fast)
        tau_slow = tau_fast + (self.max_tau - tau_fast) * torch.sigmoid(raw_gap)
        return (tau_fast + self.eps, tau_slow + self.eps)

    @staticmethod
    def _memory_retention(tau: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute continuous-time state retention ``exp(-dt / tau)``."""
        return torch.exp(-float(dt) / tau)

    def _kalman_correct(self, proposal: torch.Tensor, observation: torch.Tensor, covariance_prev: torch.Tensor, transition: torch.Tensor, process_noise: torch.Tensor, observation_noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update a state proposal and covariance using float32 Kalman arithmetic."""
        output_dtype = proposal.dtype
        with torch.amp.autocast(device_type=proposal.device.type, enabled=False):
            proposal32 = proposal.float()
            observation32 = observation.float()
            covariance_prev32 = covariance_prev.float()
            transition32 = transition.float()
            process_noise32 = process_noise.float()
            observation_noise32 = observation_noise.float()
            covariance_prior = (transition32.square() * covariance_prev32 + process_noise32).clamp(min=self.eps, max=10000.0)
            innovation = observation32 - proposal32
            innovation_covariance = (covariance_prior + observation_noise32).clamp(min=self.eps, max=10000.0)
            gain = (covariance_prior / innovation_covariance).clamp(0.0, 1.0)
            corrected = proposal32 + gain * innovation
            covariance_post = ((1.0 - gain).square() * covariance_prior + gain.square() * observation_noise32).clamp(min=self.eps, max=10000.0)
            innovation_energy = innovation.square().mean(dim=-1, keepdim=True).clamp_max(1000000.0)
            innovation_nll = 0.5 * (torch.log(innovation_covariance) + innovation_energy / innovation_covariance)
        return (corrected.to(dtype=output_dtype), covariance_post, gain.to(dtype=output_dtype), innovation.to(dtype=output_dtype), innovation_nll)

    @staticmethod
    def _zero_patch_stat(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    def _kalman_parameters(self, raw_memory: torch.Tensor, physics_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        control = torch.cat([raw_memory, physics_t], dim=-1)
        control_norm = self.kalman_control_norm(control)
        params = self.kalman_parameter_generator(control_norm)
        transition_raw = params[..., :self.state_count]
        process_raw = params[..., self.state_count:2 * self.state_count]
        observation_raw = params[..., 2 * self.state_count:3 * self.state_count]
        transition_values = torch.sigmoid(transition_raw)
        process_values = F.softplus(process_raw) + self.eps
        observation_values = F.softplus(observation_raw) + self.eps
        return (control_norm, transition_values, process_values, observation_values)

    def forward(self, x_t: torch.Tensor, physics_t: torch.Tensor, h_prev: torch.Tensor, c_fast_prev: torch.Tensor, c_slow_prev: torch.Tensor, p_fast_prev: torch.Tensor, p_slow_prev: torch.Tensor, x_fast_t: Optional[torch.Tensor]=None, x_slow_t: Optional[torch.Tensor]=None, dt: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Advance one historical frame through DTLM and PIKO correction.

        Returns the uncorrected recurrent state, corrected condition state,
        updated fast/slow memories and covariances, branch features and
        detached diagnostic statistics.
        """
        for name, tensor in {'x_t': x_t, 'physics_t': physics_t, 'h_prev': h_prev, 'c_fast_prev': c_fast_prev, 'c_slow_prev': c_slow_prev}.items():
            if tensor.ndim != 3:
                raise ValueError(f'{name} must be [B,N,D], got {tuple(tensor.shape)}')
        if x_fast_t is None:
            x_fast_t = x_t
        if x_slow_t is None:
            x_slow_t = x_t
        for name, tensor in {'x_fast_t': x_fast_t, 'x_slow_t': x_slow_t}.items():
            if tensor.ndim != 3:
                raise ValueError(f'{name} must be [B,N,D], got {tuple(tensor.shape)}')

        def _params_from(current: torch.Tensor) -> torch.Tensor:
            temporal_input = torch.cat([current, h_prev], dim=-1)
            control = current + self.temporal_fusion(temporal_input)
            return self.parameter_generator(self.parameter_norm(control))
        dynamic_params = _params_from(x_t)
        control_end = 2 * self.dim
        candidate_end = control_end + 2 * self.dim
        tau_end = candidate_end + 2 * self.dim
        control_values = dynamic_params[..., :control_end]
        output_gate, fast_mix = control_values.chunk(2, dim=-1)
        output_gate = torch.sigmoid(output_gate)
        fast_mix = torch.sigmoid(fast_mix)
        fast_params = _params_from(x_fast_t)
        slow_params = _params_from(x_slow_t)
        fast_candidates = fast_params[..., control_end:candidate_end]
        slow_candidates = slow_params[..., control_end:candidate_end]
        cand_fast, _ = fast_candidates.chunk(2, dim=-1)
        _, cand_slow = slow_candidates.chunk(2, dim=-1)
        target_fast = torch.tanh(cand_fast)
        target_slow = torch.tanh(cand_slow)
        fast_tau_values = fast_params[..., candidate_end:tau_end]
        slow_tau_values = slow_params[..., candidate_end:tau_end]
        tau_fast, _ = self._dual_time_constants(fast_tau_values)
        _, tau_slow = self._dual_time_constants(slow_tau_values)
        keep_fast = self._memory_retention(tau_fast, dt)
        keep_slow = self._memory_retention(tau_slow, dt)
        write_fast = 1.0 - keep_fast
        write_slow = 1.0 - keep_slow
        c_fast = keep_fast * c_fast_prev + write_fast * target_fast
        c_slow = keep_slow * c_slow_prev + write_slow * target_slow
        raw_fast = output_gate * torch.tanh(c_fast)
        raw_slow = output_gate * torch.tanh(c_slow)
        memory_h = fast_mix * raw_fast + (1.0 - fast_mix) * raw_slow
        physics_delta_all = self.physics_delta_proj(physics_t)
        delta_fast, delta_slow = physics_delta_all.chunk(2, dim=-1)
        physics_obs_fast = raw_fast + delta_fast
        physics_obs_slow = raw_slow + delta_slow
        zero_stat = self._zero_patch_stat(x_t)
        kalman_control, transition_values, process_values, observation_values = self._kalman_parameters(memory_h, physics_t)
        transition_fast, transition_slow = transition_values.chunk(2, dim=-1)
        process_fast, process_slow = process_values.chunk(2, dim=-1)
        observation_fast, observation_slow = observation_values.chunk(2, dim=-1)
        corrected_fast, p_fast, gain_fast, innovation_fast, nll_fast = self._kalman_correct(proposal=raw_fast, observation=physics_obs_fast, covariance_prev=p_fast_prev, transition=transition_fast, process_noise=process_fast, observation_noise=observation_fast)
        corrected_slow, p_slow, gain_slow, innovation_slow, nll_slow = self._kalman_correct(proposal=raw_slow, observation=physics_obs_slow, covariance_prev=p_slow_prev, transition=transition_slow, process_noise=process_slow, observation_noise=observation_slow)
        corrected_h = fast_mix * corrected_fast + (1.0 - fast_mix) * corrected_slow
        prior_fast = (transition_fast.square() * p_fast_prev + process_fast).detach().mean()
        prior_slow = (transition_slow.square() * p_slow_prev + process_slow).detach().mean()
        obs_noise_fast = observation_fast.detach().mean()
        obs_noise_slow = observation_slow.detach().mean()
        innovation_fast_stat = innovation_fast.detach().abs().mean()
        innovation_slow_stat = innovation_slow.detach().abs().mean()
        kalman_nll = (0.5 * (nll_fast + nll_slow)).detach().mean()
        gain_fast_stat = gain_fast.detach().mean()
        gain_slow_stat = gain_slow.detach().mean()
        physics_delta_abs = 0.5 * (delta_fast.detach().abs().mean() + delta_slow.detach().abs().mean())
        correction_abs = (corrected_h.detach() - memory_h.detach()).abs().mean()
        stats = {'tau_fast': tau_fast.detach().mean(), 'tau_slow': tau_slow.detach().mean(), 'fast_mix': fast_mix.detach().mean(), 'raw_memory_abs': memory_h.detach().abs().mean(), 'physics_delta_abs': physics_delta_abs, 'physics_correction_abs': correction_abs, 'kalman_gain_fast': gain_fast_stat, 'kalman_gain_slow': gain_slow_stat, 'prior_uncertainty_fast': prior_fast, 'prior_uncertainty_slow': prior_slow, 'observation_noise_fast': obs_noise_fast, 'observation_noise_slow': obs_noise_slow, 'innovation_fast': innovation_fast_stat, 'innovation_slow': innovation_slow_stat, 'kalman_nll': kalman_nll}
        branch_state = {'fast': corrected_fast, 'slow': corrected_slow, 'tau_fast': tau_fast, 'tau_slow': tau_slow}
        return (memory_h, corrected_h, c_fast, c_slow, p_fast, p_slow, branch_state, stats)
