import os
import random
import shutil
from pathlib import Path
import fire


def create_dirs(out_dir, splits):
    out_dir = Path(out_dir)

    for split in splits:
        for sub in ["images", "labels"]:
            path = out_dir / split / sub
            path.mkdir(parents=True, exist_ok=True)


def split_dataset(data_dir, out_dir, train=0.7, val=0.15):
    train_val_dirs = ["Train", "Val"]
    tumor_dirs = [
        d
        for d in os.listdir(Path(data_dir) / train_val_dirs[0])
        if not d.startswith(".")
    ]

    splits = ["train", "val", "test"]
    create_dirs(out_dir, splits)

    for tumor_dir in tumor_dirs:
        for train_val_dir in train_val_dirs:
            img_dir = Path(data_dir) / train_val_dir / tumor_dir / "images"
            files = list(img_dir.glob("*.jpg"))

            random.seed(42)
            random.shuffle(files)

            n = len(files)
            n_train = int(n * train)
            n_val = int(n * val)

            train_files = files[:n_train]
            val_files = files[n_train : n_train + n_val]
            test_files = files[n_train + n_val :]

            def copy_images(data_dir, train_val_dir, tumor_dir, file_names, split):
                for file_name in sorted(file_names):
                    old_path = Path(data_dir) / train_val_dir / tumor_dir

                    new_path = Path(out_dir) / split

                    path = new_path / "images"
                    num_files = str(len(list(path.glob("*.jpg"))) + 1)
                    new_file_name = num_files + tumor_dir[0] + split[0:2]

                    shutil.copy2(file_name, path / f"{new_file_name}.jpg")

                    label_dst = new_path / "labels" / f"{new_file_name}.txt"

                    if tumor_dir == "No Tumor":
                        label_dst.write_text("")

                    else:
                        label_path = old_path / "labels" / f"{file_name.stem}.txt"
                        shutil.copy2(label_path, label_dst)

            copy_images(data_dir, train_val_dir, tumor_dir, train_files, "train")
            copy_images(data_dir, train_val_dir, tumor_dir, val_files, "val")
            copy_images(data_dir, train_val_dir, tumor_dir, test_files, "test")


if __name__ == "__main__":
    fire.Fire(split_dataset)
