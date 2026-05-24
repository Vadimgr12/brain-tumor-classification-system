from pathlib import Path
import torch

from model_loader import (
    load_brain_tumor_classification_model,
    resolve_device,
)
from omegaconf import DictConfig
import hydra


def get_best_checkpoint(checkpoint_dir: str | Path):

    checkpoint_dir = Path(checkpoint_dir)
    all_checkpoints = list(checkpoint_dir.glob("*best*.ckpt"))

    return max(all_checkpoints, key=lambda p: p.stat().st_mtime)


@hydra.main(config_path="../conf", config_name="infer", version_base="1.3")
def export_onnx(cfg: DictConfig) -> None:

    checkpoint_path = get_best_checkpoint("checkpoints")
    tumor_classification_project_dir = cfg.paths.model_dir
    device = resolve_device(cfg.model.device)
    model = load_brain_tumor_classification_model(
        tumor_classification_project_dir, checkpoint_path, device
    )
    model.eval()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    onnx_path = artifacts_dir / "model.onnx"

    image_size = cfg.data.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    with torch.inference_mode():
        torch.onnx.export(model, dummy_input, str(onnx_path), opset_version=18)

    print("Successfully saved in ONNX!")


if __name__ == "__main__":
    export_onnx()
