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
    BinaryConfusionMatrix
)
from torch.optim.lr_scheduler import CosineAnnealingLR


class BrainTumorModule(L.LightningModule):

    def __init__(self,n_unfrozen ,lr, t_max_scheduler,weight_decay,out_classes, no_tumor_class):

        super().__init__()

        self.model = get_model(n_unfrozen, out_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.t_max_scheduler = t_max_scheduler
        self.weight_decay = weight_decay
        self.no_tumor_class = no_tumor_class

        self.val_acc = Accuracy(task="multiclass", num_classes=out_classes)
        self.val_recall = Recall(task="multiclass", num_classes=out_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=out_classes, average="macro")
        self.val_bin_recall = BinaryRecall()
        self.val_pr_auc = AveragePrecision(task="multiclass",num_classes=out_classes ,average="macro")
        self.bin_cm = BinaryConfusionMatrix()


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def to_binary(self, preds, targets):
        preds_bin = preds != self.no_tumor_class
        targets_bin = targets != self.no_tumor_class

        return preds_bin.int(), targets_bin.int()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        preds_bin, y_bin = self.to_binary(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_acc.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_pr_auc.update(probs,y)
        self.val_bin_recall.update(preds_bin, y_bin)
        self.bin_cm.update(preds_bin, y_bin)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute())
        self.log("val_recall", self.val_recall.compute())
        self.log("val_f1", self.val_f1.compute())
        self.log("pr_auc", self.val_pr_auc.compute())
        cm = self.bin_cm.compute()
        tn, fp, fn, tp = cm.ravel()

        self.log("val_recall_bin", self.val_bin_recall.compute())
        self.log("binary_fn", fn.float())
        self.val_acc.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_bin_recall.reset()
        self.bin_cm.reset()
        self.val_pr_auc.reset()

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
            #filter(lambda p: p.requires_grad, self.parameters()),
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




