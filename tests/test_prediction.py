import numpy as np
from polastflow.data import RadarPredictionDataset


def test_prediction_dataset_needs_no_future_labels(tmp_path):
    folder = tmp_path / 'input';folder.mkdir()
    a = np.zeros((2, 2, 32, 32), dtype='float32')
    a[0] = 35
    a[1] = 0.5
    np.save(folder / 'input_001.npy', a)
    dataset = RadarPredictionDataset(str(tmp_path), history_frames=2, img_size=32)
    x, name = dataset[0]
    assert x.shape == (2, 2, 32, 32)
    assert float(x[0].mean()) == 0.5
    assert abs(float(x[1].mean()) - 0.3) < 1e-6
    assert name == 'input_001.npy'
