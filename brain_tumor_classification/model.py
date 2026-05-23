import lightning as L
import torch
import torch.nn as nn
from utilities.model_getter import get_model
# from torchvision.models import (
#
#     efficientnet_v2_s,
#
#     EfficientNet_V2_S_Weights
#
# )

class BrainTumorModule(L.LightningModule):

    def __init__(self, lr, out_classes):

        super().__init__()

        # self.model = efficientnet_v2_s(
        #
        #     weights=EfficientNet_V2_S_Weights.DEFAULT
        #
        # )
        #
        # in_features = self.model.classifier[1].in_features
        #
        # self.model.classifier[1] = nn.Linear(in_features, 4)
        #
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        #
        # for param in self.model.features[-2:].parameters():
        #     param.requires_grad = True
        #
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        self.model = get_model(out_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):

        return torch.optim.AdamW(

            filter(lambda p: p.requires_grad, self.parameters()),

            lr=self.lr

        )




