# Data format

Training/validation dataset folders each contain `input/input_*.npy` and the matching `output/output_*.npy` files. Prediction input folders need only `input/input_*.npy`.

For default configuration, array shape is `[2, 10, 256, 256]` with channels `ZH`, `ZDR`. The source loader verifies the exact dimensions and aligns corresponding files by replacing the `input_` basename prefix with `output_`.

The normalization is inherited from the original implementation:

- `ZH`: clip to `[0, 70]` then divide by 70.
- `ZDR`: clip to `[-1, 4]` then calculate `(ZDR + 1) / 5`.
- First replace NaN, positive infinity and negative infinity with zero before clipping.

Predictions are saved as channel-first NumPy arrays in model space, without clipping or inverse conversion. They are not automatically converted to dBZ or dB.
