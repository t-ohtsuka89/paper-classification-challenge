import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.classification.f_beta import FBetaScore

from model import BaseModel


class LightningModel(pl.LightningModule):
    def __init__(self, model_name: str, border):
        super().__init__()
        self.model = BaseModel(model_name)
        self.criterion = self.create_criterion()
        self.fbeta = FBetaScore(beta=0.7, threshold=border, num_classes=1)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        input_ids: Tensor
        attention_mask: Tensor
        labels: Tensor

        y_preds = self.model(input_ids, attention_mask)
        loss: torch.Tensor = self.criterion(y_preds, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        input_ids: Tensor
        attention_mask: Tensor
        labels: Tensor

        y_preds: torch.Tensor = self.model(input_ids, attention_mask)
        loss: torch.Tensor = self.criterion(y_preds, labels)

        self.fbeta(y_preds, labels.to(dtype=torch.int64))
        self.log("val_loss", loss)
        self.log("val_fbeta-score", self.fbeta, on_step=False, on_epoch=True)

        return y_preds, labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer

    def create_criterion(self):
        criterion = nn.BCELoss()
        return criterion
