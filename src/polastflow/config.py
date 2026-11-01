from pathlib import Path
from types import SimpleNamespace
import yaml


def load_model_config(path: str | Path):
    """Only the architecture and data normalization, never a training preset."""
    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('model config must be a YAML mapping')
    return SimpleNamespace(**config)
