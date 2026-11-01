from types import SimpleNamespace
import torch
from polastflow.engine.checkpoint import save_checkpoint, load_training_checkpoint


def test_checkpoint_resume(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    p = tmp_path / 'last.pth'
    save_checkpoint(str(p), model, optimizer, scaler, 2, 0.25,
                    SimpleNamespace(hidden_size=384), SimpleNamespace(epochs=5))
    start, best = load_training_checkpoint(str(p), model, optimizer, scaler, 'cpu')
    assert (start, best) == (3, 0.25)
