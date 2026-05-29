import lightning as L
import hydra
from data.data import BrainTumorDataModule
from model import BrainTumorModule
from data.transforms import get_train_transforms, get_val_transforms
from omegaconf import DictConfig
import json
from utilities.mlflow_logger_getter import build_logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:

    with open("data/stats/mean_std.json") as f:
        stats = json.load(f)

    dataset_mean = stats["mean"]
    dataset_std = stats["std"]

    datamodule = BrainTumorDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
        train_transform=get_train_transforms(cfg.transform, dataset_mean, dataset_std),
        val_transform=get_val_transforms(cfg.transform, dataset_mean, dataset_std),
    )

    model = BrainTumorModule(
        n_unfrozen=cfg.model.n_unfrozen,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        t_max_scheduler=cfg.training.max_epochs,
        out_classes=cfg.training.out_classes,
        no_tumor_class=cfg.training.no_tumor_class,
    )

    early_stopping = EarlyStopping(
        monitor="Recall",
        mode="max",
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="best",
        monitor="Recall",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        early_stopping,
    ]

    logger = build_logger(
        cfg.logger.mlflow.tracking_uri,
        cfg.logger.mlflow.experiment_name,
        cfg.logger.mlflow.run_name,
    )

    trainer = L.Trainer(
        default_root_dir="runs/",
        accelerator=cfg.training.accelerator,
        devices=cfg.training.num_devices,
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    logger.experiment.log_artifact(
        logger.run_id,
        checkpoint_callback.best_model_path,
        artifact_path="checkpoints",
    )


if __name__ == "__main__":
    train()
