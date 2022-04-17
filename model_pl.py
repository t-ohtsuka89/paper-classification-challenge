import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import fbeta_score
from torch import Tensor, nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from torchmetrics.classification.f_beta import FBetaScore


class LightningModel(pl.LightningModule):
    def __init__(self, model_name: str, border):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = self.create_criterion()
        self.border = border
        self.fbeta = FBetaScore(beta=0.7, threshold=border, num_classes=1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        out: SequenceClassifierOutput = self.model(input_ids=input_ids, attention_mask=attention_mask)
        preds: Tensor = self.sigmoid(out.logits)
        return preds.squeeze()

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        input_ids: Tensor
        attention_mask: Tensor
        labels: Tensor

        y_preds = self.forward(input_ids, attention_mask)
        y_preds = y_preds.view(
            -1,
        )
        loss: torch.Tensor = self.criterion(y_preds, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        input_ids: Tensor
        attention_mask: Tensor
        labels: Tensor

        y_preds = self.forward(input_ids, attention_mask)
        y_preds = y_preds.view(
            -1,
        )
        loss = self.criterion(y_preds, labels)
        self.fbeta(y_preds, labels.to(dtype=torch.int64))
        self.log("val_loss", loss)
        self.log("val_fbeta-score", self.fbeta, on_step=False, on_epoch=True)

        return y_preds, labels

    # def validation_epoch_end(self, outputs) -> None:
    #     all_preds = []
    #     all_labels = []
    #     for output in outputs:
    #         preds, labels = output
    #         all_preds.append(preds)
    #         all_labels.append(labels)
    #     all_preds = torch.stack(all_preds).cpu()
    #     all_labels = torch.stack(all_labels).cpu()
    #     score = fbeta_score(all_labels, np.where(
    #         all_preds < self.border, 0, 1), beta=7.0)
    #     self.log("val_fbeta_score", score)
    #     return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer

    def create_criterion(self):
        criterion = nn.BCELoss()
        return criterion
