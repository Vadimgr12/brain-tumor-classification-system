import lightning as L
import torch
import torch.nn as nn
from utilities.model_getter import get_model
from torchmetrics.classification import (
    Accuracy,
    Recall,
    F1Score,
    BinaryRecall,
    AveragePrecision,
    ConfusionMatrix,
    AUROC,
)
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


class BrainTumorModule(L.LightningModule):
    def __init__(
        self,
        n_unfrozen: int,
        lr: float,
        t_max_scheduler: int,
        weight_decay: float,
        out_classes: int,
        no_tumor_class: int,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(n_unfrozen, out_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.t_max_scheduler = t_max_scheduler
        self.weight_decay = weight_decay
        self.no_tumor_class = no_tumor_class

        self.acc = Accuracy(task="multiclass", num_classes=out_classes)
        self.recall = Recall(
            task="multiclass", num_classes=out_classes, average="macro"
        )
        self.f1 = F1Score(task="multiclass", num_classes=out_classes, average="macro")
        self.bin_recall = BinaryRecall()
        self.pr_auc = AveragePrecision(
            task="multiclass", num_classes=out_classes, average="macro"
        )
        self.auroc = AUROC(task="multiclass", num_classes=out_classes, average="macro")
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def to_binary(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_bin = preds != self.no_tumor_class
        targets_bin = targets != self.no_tumor_class

        return preds_bin.int(), targets_bin.int()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        preds_bin, y_bin = self.to_binary(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.acc.update(preds, y)
        self.recall.update(preds, y)
        self.f1.update(preds, y)
        self.pr_auc.update(probs, y)
        self.bin_recall.update(preds_bin, y_bin)
        self.auroc.update(probs, y)
        self.confmat.update(preds, y)

        return loss

    def on_validation_epoch_end(self):
        self.log("Accuracy", self.acc.compute())
        self.log("Recall", self.recall.compute())
        self.log("F1", self.f1.compute())
        self.log("PR-AUC", self.pr_auc.compute())
        self.log("ROC-AUC", self.auroc.compute())

        labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
        order = [2, 0, 1, 3]
        confmat_counted = self.confmat.compute().cpu().numpy()
        confmat_counted_ordered = confmat_counted[order][:, order]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confmat_counted_ordered,
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
        )

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()

        self.logger.experiment.log_figure(
            run_id=self.logger.run_id,
            figure=fig,
            artifact_file=f"confusion_matrix_epoch_{self.current_epoch:03d}.png",
        )

        plt.close(fig)

        self.log("Recall_binary", self.bin_recall.compute())
        self.acc.reset()
        self.recall.reset()
        self.f1.reset()
        self.bin_recall.reset()
        self.pr_auc.reset()
        self.auroc.reset()
        self.confmat.reset()

    def configure_optimizers(self):
        decay, no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bn" in name.lower() or "bias" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            # filter(lambda p: p.requires_grad, self.parameters()),
            [
                {"params": decay, "weight_decay": 1e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.t_max_scheduler,
            eta_min=self.lr * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
