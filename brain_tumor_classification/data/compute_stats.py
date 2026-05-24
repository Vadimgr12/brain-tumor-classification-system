from pathlib import Path
from PIL import Image
import numpy as np
import json
from argparse import ArgumentParser
import random


def compute_train_stats(data_dir, out_path):
    image_paths = sorted(list((Path(data_dir) / "images").glob("*.jpg")))
    random.seed(42)
    image_paths = random.sample(image_paths, int(0.9 * len(image_paths)))

    sum_ch = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)

    n_pixels = 0

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image) / 255.0

        sum_ch += image.sum(axis=(0, 1))
        sum_sq += (image**2).sum(axis=(0, 1))
        n_pixels += image.shape[0] * image.shape[1]

    mean = sum_ch / n_pixels
    std = np.sqrt(sum_sq / n_pixels - mean**2)

    stats = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    return mean, std


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    compute_train_stats(args.input, args.output)
