from typing import Union
import numpy as np
import torch
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io


def get_val_transforms(mean, std, image_size=224):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def preprocess(x: Union[bytes, np.ndarray, torch.Tensor]):
    if isinstance(x, bytes):
        x = Image.open(io.BytesIO(x)).convert("RGB")
        x = np.array(x)

    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    with open("data/stats/mean_std.json") as f:
        stats = json.load(f)

    dataset_mean = stats["mean"]
    dataset_std = stats["std"]
    transform = get_val_transforms(dataset_mean, dataset_std, image_size=224)
    x = transform(image=x)["image"]

    x = np.expand_dims(x, axis=0)
    return x
