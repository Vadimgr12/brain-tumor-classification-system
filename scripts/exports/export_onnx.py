from pathlib import Path
import torch
from get_module import load_model_module
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def export_onnx(cfg: DictConfig) -> None:

    checkpoint_path = cfg.infer.paths.checkpoint_path

    tumor_classification_project_dir = cfg.infer.paths.model_dir

    model_loader = load_model_module(cfg.infer.paths.module_dir)
    device = model_loader.resolve_device(cfg.infer.model.device)
    model = model_loader.load_brain_tumor_classification_model(
        tumor_classification_project_dir, checkpoint_path, device
    )

    model.eval()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    onnx_path = artifacts_dir / "model.onnx"

    image_size = cfg.data.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=17,
            input_names=["x"],
            output_names=["logits"],
        )

    print("Successfully saved in ONNX!")


if __name__ == "__main__":
    export_onnx()
