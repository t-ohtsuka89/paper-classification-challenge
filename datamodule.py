import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from dataset import BaseDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, fold: int, tokenizer: PreTrainedTokenizer, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.fold = fold
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None):

        data_dir = "data"
        folds = pd.read_csv(os.path.join(data_dir, "train_folds.csv"))
        train_fold = folds.loc[folds["kfold"] != self.fold]
        val_fold = folds.loc[folds["kfold"] == self.fold]

        self.train_ds = BaseDataset(train_fold, self.tokenizer)
        self.val_ds = BaseDataset(val_fold, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
