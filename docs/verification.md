# Refactoring verification report

## Provenance

Source of truth: the user-supplied archive **`新建文件夹(1).zip`**. No previously generated/open-source ZIP was used as the source implementation.

The full-model reference uses all four original ablation options set to `True`. In this repository the full computation is unconditional; no public ablation options or disabled alternatives remain.

## Original-to-new source mapping

| Original source | New source |
| --- | --- |
| `models/spatial_encoder.py` | `models/psr/spectral.py`, `blocks.py`, `encoder.py` |
| `models/physics_encoder.py` | `models/piko/encoder.py` |
| `models/temporal_memory.py` | `models/dtlm/cell.py`, `block.py`, `memory.py` |
| `models/history_encoder.py` | `models/condition.py` |
| `models/pmf_blocks.py` | `models/generation/rope.py`, `attention.py`, `blocks.py` |
| `models/pmf_sequence_backbone.py` | `models/generation/backbone.py` |
| `models/pixel_mean_flow.py` | `models/generation/flow.py` |
| `models/layers.py` | `models/common/layers.py` |
| `models/patch_layers.py` | `models/common/patch.py` |
| `model_factory.py` | `models/factory.py` |
| `dataset.py` | `data/dataset.py` (original paired dataset retained; prediction-only dataset appended) |
| `evaluate.py` | `engine/validation.py` (original validation raw loss; forecast metrics deleted) |
| `train.py`, `utils.py`, `test.py` | `engine/` and `scripts/` (generic training, checkpoint, prediction) |

## Class names

| Original class | New class |
| --- | --- |
| `ConditionalPixelMeanFlow` | `PolaSTFlow` |
| `RadarHistoryEncoder` | `HistoricalConditionEncoder` |
| `PolarimetricSpectralParsingEncoder` | `PolarimetricSpectralRepresentation` |
| `DualPolPhysicsEncoder` | `PhysicsInformedObservationEncoder` |
| `MotionAwareTemporalEncoder` | `DualTimescaleLiquidMemory` |
| `PhysicsGuidedLiquidCell` | `KalmanGuidedLiquidCell` |
| `PhysicsGuidedTemporalBlock` | `LiquidMemoryBlock` |
| `PMFSequenceBackbone` | `PixelMeanFlowSequenceBackbone` |

## Full-path ablation cleanup

- Spatial and physics encoder construction now always takes the original full-model path.
- PSR three-band tokens always generate fused, fast and slow features.
- Liquid memory always uses two timescales and Kalman observation correction.
- Single-timescale and no-Kalman alternative forward paths have been removed.
- **No ablation options are exposed** in model YAML, constructors or training CLI.
- Original unused `direct_physics_gate` parameter tensors and the original `frame_patch_embed` in historical encoding were retained for checkpoint/parameter-name compatibility, not for switching model paths.

## Numeric comparison against original source

| Test | Original | Refactored | Observation |
| --- | --- | --- | --- |
| Default model parameters | 17,472,882 | 17,472,882 | Equal |
| Default trainable parameters | 16,686,450 | 16,686,450 | Equal |
| Default `state_dict` keys | 312 | 312 | Exact key order/shape match |
| Default model initialization, seed 42 | source model | release model | Tensor-wise max difference **0** |
| Reduced model parameters | 520,126 | 520,126 | Equal |
| Reduced model state keys | 288 | 288 | Equal |
| Reduced-model MeanFlow loss | 1.9999980926513672 | 1.9999980926513672 | Difference **0** |
| Reduced-model `raw_total` | 5.032339096069336 | 5.032339096069336 | Difference **0** |
| Reduced-model parameter gradients | source model | release model | Maximum difference **0** |
| Reduced-model one-step prediction | source model | release model | Maximum difference **0** |
| Original `evaluate.validation_loss` vs new `engine.validation.validation_loss` | 2.221303343773 | 2.221303343773 | Difference **0** |

The reduced model test used `[B,C,T,H,W]=[1,2,2,32,32]`, 64 hidden channels, 3 prediction blocks, identical initial `state_dict`, and explicitly identical time and Gaussian noise samples. These tests verify computation on this concrete case, not proof of universal bitwise equivalence across every CUDA kernel/hardware combination.

## Training and prediction smoke tests

In the available CPU/PyTorch 2.10 environment:

- Python compile check: **passed**.
- Unit tests (`pytest -q`): **4 passed**.
- Forward/backward/one-step self-check: **passed**.
- Tiny dataset, first epoch: `best.pth` and `last.pth` both created.
- Resume using `last.pth`, train second epoch: **passed**.
- Resume retained the previous minimum validation loss and refreshed `best.pth` when an improvement occurred.
- Inference without target files: generated two `.npy` files of shape `[2,2,32,32]`.
- Editable package install: **passed** with offline command `pip install -e . --no-build-isolation --no-deps`, using already-installed dependencies.
- Standard isolated editable install: **not verified** because sandbox pip could not reach package indexes to fetch isolated build requirements; this is not evidence of a source defect.

## Behavior retained or intentionally changed

- Training continues to backpropagate original normalized MeanFlow loss, not the `raw_total` logging quantity.
- Optimizer: original `AdamW`, `(0.9, 0.95)` betas, `1e-4` default LR and weight decay, and `0.3` gradient clip.
- Original non-finite-loss and gradient batch-skip handling retained.
- Training startup seed `42 + rank`; **never** reset each batch or epoch.
- Original validation loss uses sample-weighted aggregation and one `torch.Generator(...).manual_seed(12345)` per complete evaluation pass.
- Inference generator is initialized once per complete prediction pass at `12345`.
- Checkpoint naming and selection deliberately changed: only `best.pth` on minimum validation `raw_total`, plus per-epoch `last.pth`.
- Forecast metrics and labeled-test evaluation deliberately deleted.
- No augmentation, filtering or new loss introduced.
- The old `num_workers=0`, default batch 8, validation batch 4, 200 epochs, and AMP-disabled defaults are retained.

## Outstanding release checks

- **CUDA/NCCL multi-GPU DDP training and full-resolution 256 × 256 GPU training were not run** in this CPU-only verification environment.
- DDP resume does not guarantee identical per-rank random-number streams (the rank 0 checkpoint cannot capture RNG states of every other rank). It does preserve distinct rank seeds.
- The `LICENSE` file is currently a placeholder; copyright-holder approval and third-party code license review are required before public open-source distribution.
- This repository does not make any claim about newly reproduced TGRS forecast metrics; metrics are intentionally not computed.

## Model documentation cleanup

- Replaced outdated PSP and Temporal Drifting descriptions with PSR, PIKO, DTLM and Pixel MeanFlow terminology.
- Removed the fully commented-out, disabled duplicate implementation at the top of the PIKO encoder file.
- No executable Python syntax trees, model keys, optimization steps or model behavior were changed.
- All model Python files were compared after stripping only Docstrings; executable AST matches the prior release exactly.
- The pre-existing runtime error message strings and diagnostic dictionary keys (including `psp_` keys) were deliberately retained for interface compatibility.
