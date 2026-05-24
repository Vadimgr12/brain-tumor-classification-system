from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import lightning as L
import torch
import random
import albumentations as A
import numpy as np


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_samples: int | None = None,
        transform: A.Compose | None = None,
    ):
        self.data_dir = Path(data_dir)

        self.image_paths = sorted(list((self.data_dir / "images").glob("*.jpg")))

        self.transform = transform

        if max_samples is not None:
            self.image_paths = random.sample(self.image_paths, max_samples)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        label_path = self.data_dir / "labels" / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = self.transform(image=image)
        image = image["image"]

        with open(label_path) as f:
            first_line = f.readline().strip()

        if first_line == "":
            class_id = 3

        else:
            class_id = int(first_line.split()[0])

        return image, torch.tensor(class_id, dtype=torch.long)


class BrainTumorDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        max_samples,
        train_transform=None,
        val_transform=None,
    ):

        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = BrainTumorDataset(
                f"{self.hparams.data_dir}/train",
                max_samples=self.hparams.max_samples,
                transform=self.hparams.train_transform,
            )

            self.val_dataset = BrainTumorDataset(
                f"{self.hparams.data_dir}/val",
                max_samples=self.hparams.max_samples,
                transform=self.hparams.val_transform,
            )
        if stage in ("test", None):
            self.test_dataset = BrainTumorDataset(
                f"{self.hparams.data_dir}/test",
                max_samples=self.hparams.max_samples,
                transform=self.hparams.val_transform,
            )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )
