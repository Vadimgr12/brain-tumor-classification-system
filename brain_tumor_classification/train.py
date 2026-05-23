import lightning as L
import hydra
from data.data import BrainTumorDataModule
from model import BrainTumorModule
from data.transforms import get_train_transforms, get_val_transforms
from omegaconf import DictConfig
import json

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:

    with open("data/stats/mean_std.json") as f:
        stats = json.load(f)

    dataset_mean = stats["mean"]
    dataset_std = stats["std"]

    datamodule = BrainTumorDataModule(
        data_dir = cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
        train_transform=get_train_transforms(
            cfg.transform, dataset_mean, dataset_std
        ),
        val_transform=get_val_transforms(
            cfg.transform, dataset_mean, dataset_std
        )
    )

    model = BrainTumorModule(
        lr=cfg.training.lr,
        out_classes=cfg.model.out_classes
    )

    trainer = L.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=10,
        precision="32",
    )

    trainer.fit(
        model,
        datamodule=datamodule
    )


if __name__ == "__main__":
    train()

