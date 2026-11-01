from types import SimpleNamespace
from pathlib import Path
import torch
from polastflow.config import load_model_config
from polastflow.models.factory import build_model


def test_full_model_forward_backward_and_sampling():
    cfg = vars(load_model_config(Path(__file__).resolve().parents[1] / 'configs/model.yaml'))
    cfg.update(img_size=32, history_frames=2, future_frames=2, hidden_size=64,
               prediction_depth=3, pmf_aux_head_depth=1, num_heads=4,
               bottleneck_dim=32, condition_bottleneck_dim=24,
               pmf_interval_tokens=2)
    model = build_model(SimpleNamespace(**cfg))
    x = torch.rand(1, 2, 2, 32, 32)
    y = torch.rand_like(x)
    loss, logs = model(x, y, return_logs=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(logs['raw_total'])
    loss.backward()
    model.eval()
    with torch.no_grad():
        z = model.generate(x, generator=torch.Generator().manual_seed(12345))
    assert z.shape == y.shape


def test_no_public_ablation_arguments():
    import inspect
    from polastflow.models.condition import HistoricalConditionEncoder
    from polastflow.models.dtlm.cell import KalmanGuidedLiquidCell
    for obj in (HistoricalConditionEncoder, KalmanGuidedLiquidCell):
        signature = str(inspect.signature(obj.__init__))
        assert 'use_spatial_encoder' not in signature
        assert 'dual_timescale' not in signature
        assert 'use_kalman' not in signature
