import importlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device_name: str) -> torch.device:

    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device_name)


def resolve_repo_path(path_like: str | Path) -> Path:

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _append_brain_tumor_classification_dir(cityscapes_project_dir: Path) -> Path:
    if str(cityscapes_project_dir) not in sys.path:
        sys.path.insert(0, str(cityscapes_project_dir))
    return cityscapes_project_dir


def load_lightning_module(tumor_classification_project_dir: str | Path):
    """Import the Cityscapes Lightning module after adding its local path."""
    _append_brain_tumor_classification_dir(
        resolve_repo_path(tumor_classification_project_dir)
    )
    return importlib.import_module("model")


def load_brain_tumor_classification_model(
    tumor_classification_project_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load the trained Cityscapes Lightning checkpoint."""
    model_module = load_lightning_module(tumor_classification_project_dir)
    model = model_module.BrainTumorModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
        n_unfrozen=3,  # ⚠️ если у тебя есть — поставь реальное значение
        lr=1e-3,
        t_max_scheduler=15,  # обычно = max_epochs
        weight_decay=1e-4,
        out_classes=4,
        no_tumor_class=2,
    )
    model.eval()
    return model.to(device)
