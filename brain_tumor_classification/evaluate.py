import torch
from pathlib import Path
from data.data import BrainTumorDataset
from omegaconf import DictConfig
from data.transforms import get_val_transforms
from torch.utils.data import DataLoader
from utilities.model_loader import load_brain_tumor_classification_model
import json
import hydra
from torchmetrics.classification import (
    Accuracy,
    Recall,
    F1Score,
    AveragePrecision,
    AUROC,
    ConfusionMatrix,
    BinaryRecall,
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_device(cfg):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def eval(cfg: DictConfig):
    with open("data/stats/mean_std.json") as f:
        stats = json.load(f)

    dataset_mean = stats["mean"]
    dataset_std = stats["std"]

    test_dataset = BrainTumorDataset(
        data_dir=cfg.eval.data.data_dir,
        transform=get_val_transforms(cfg.transform, dataset_mean, dataset_std),
    )

    loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.data.batch_size,
        shuffle=False,
        num_workers=cfg.eval.data.num_workers,
    )

    device = get_device(cfg.eval)

    model = load_brain_tumor_classification_model(
        cfg.eval.model.model_dir, cfg.eval.checkpoint.path, device
    )

    out_classes = cfg.eval.model.out_classes

    acc = Accuracy(task="multiclass", num_classes=out_classes).to(device)
    recall = Recall(task="multiclass", num_classes=out_classes, average="macro").to(
        device
    )
    f1 = F1Score(task="multiclass", num_classes=out_classes, average="macro").to(device)
    pr_auc = AveragePrecision(
        task="multiclass", num_classes=out_classes, average="macro"
    ).to(device)
    auroc = AUROC(task="multiclass", num_classes=out_classes, average="macro").to(
        device
    )
    confmat = ConfusionMatrix(task="multiclass", num_classes=out_classes).to(device)
    bin_recall = BinaryRecall().to(device)

    model.eval()

    no_tumor_class = cfg.eval.model.no_tumor_class
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            preds_bin = (preds != no_tumor_class).int()
            targets_bin = (y != no_tumor_class).int()

            bin_recall.update(preds_bin, targets_bin)
            acc.update(preds, y)
            recall.update(preds, y)
            f1.update(preds, y)
            pr_auc.update(probs, y)
            auroc.update(probs, y)
            confmat.update(preds, y)

    results = {
        "accuracy": acc.compute().item(),
        "recall": recall.compute().item(),
        "f1": f1.compute().item(),
        "pr_auc": pr_auc.compute().item(),
        "roc_auc": auroc.compute().item(),
        "binary_recall": bin_recall.compute().item(),
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    cm = confmat.compute().cpu().numpy()
    labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    order = [2, 0, 1, 3]
    cm = cm[order][:, order]
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")


if __name__ == "__main__":
    eval()
