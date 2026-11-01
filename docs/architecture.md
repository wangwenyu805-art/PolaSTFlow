# Model organization

This repository implements the full PolaSTFlow sequence predictor using the original Tensor operations and original parameter names.

The model comprises three history-condition processing stages and one Pixel MeanFlow generation stage:

1. `models/psr/`: Gaussian spatial frequency decomposition into three bands, common/differential polarimetric features, and band refinement.
2. `models/piko/`: physical evolution features and observation tokens.
3. `models/dtlm/`: data-driven fast/slow recurrent memory and Kalman-based physics observation correction. The `cell.py`, `block.py` and `memory.py` files originate from three classes in the original `models/temporal_memory.py`.
4. `models/generation/`: joint future-frame trajectory patching, pMF RoPE and transformer blocks, instantaneous and average velocity prediction, original MeanFlow/JVP objective and one-step generation.

The main network class is `PolaSTFlow`, constructed via `polastflow.models.factory.build_model`. The complete condition encoder is `HistoricalConditionEncoder`.

The original disabled-module branches were removed by selecting the path taken when each of the four original switches was `True`. Two originally allocated weight tensors in `direct_physics_gate` remain for strict checkpoint compatibility, even though that historical no-Kalman branch is not invoked by the full-model forward path. This does not expose an ablation option.
