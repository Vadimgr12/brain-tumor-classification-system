import hydra
from omegaconf import DictConfig
from pathlib import Path


def build_config(cfg: DictConfig) -> str:
    return f"""
name: "{cfg.model.name}"
platform: "{cfg.model.platform}"

input [
  {{
    name: "{cfg.input.name}"
    data_type: {cfg.input.dtype}
    dims: {list(cfg.input.dims)}
  }}
]

output [
  {{
    name: "{cfg.output.name}"
    data_type: {cfg.output.dtype}
    dims: {list(cfg.output.dims)}
  }}
]
""".strip()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    config_text = build_config(cfg.triton)

    model_dir = Path("triton/models") / cfg.triton.model.name
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config_text)

    print(f"Saved Triton config to: {config_path}")


if __name__ == "__main__":
    main()
