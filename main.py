import math
import os
import time
import warnings
from argparse import ArgumentParser
from logging import Logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from dataset import BaseDataset
from model import BaseModel
from model_pl import LightningModel
from utils.seed import seed_everything

warnings.filterwarnings("ignore")


def get_kfold_data(data_frame: pd.DataFrame, fold_num: int, random_state: int) -> pd.DataFrame:
    sfk = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=random_state)
    for n, (_, val_index) in enumerate(sfk.split(data_frame, data_frame["judgement"])):
        data_frame.loc[val_index, "fold"] = int(n)
    data_frame["fold"] = data_frame["fold"].astype(np.uint8)
    return data_frame


def train_loop(
    train: pd.DataFrame,
    fold: int,
    epochs: int,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    border: float,
    device: torch.device,
    output_dir: str,
):
    print(f"========== fold: {fold} training ==========")

    trn_idx = train[train["fold"] != fold].index
    val_idx = train[train["fold"] == fold].index

    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)

    train_dataset = BaseDataset(train_folds, tokenizer)
    valid_dataset = BaseDataset(valid_folds, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = LightningModel(model_name, border)
    model.to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir),
        filename=f"model_{fold:02d}",
        monitor="val_fbeta-score",
        verbose=False,
        save_last=False,
        save_top_k=1,
        save_weights_only=False,
        mode="max",
    )
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def get_result(result_df: pd.DataFrame, border: float, logger: Logger):
    preds = result_df["preds"].to_numpy()
    labels = result_df["judgement"].to_numpy()
    score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
    logger.info(f"Score: {score:<.5f}")


def main(hparams):
    data_dir = hparams.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = hparams.epochs
    fold_size: int = hparams.fold
    model_name: str = hparams.model_name
    output_dir: str = hparams.output_dir
    seed = hparams.seed

    if not os.path.exists(data_dir):
        raise FileNotFoundError("Not found data")

    # make directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # fix seed
    seed_everything(seed)

    # load data
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "sample_submit.csv"), header=None)
    sub.columns = ["id", "judgement"]

    train["judgement"][2488] = 0
    train["judgement"][7708] = 0
    border = len(train[train["judgement"] == 1]) / len(train["judgement"])
    train = get_kfold_data(train, fold_size, seed)

    # model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training
    for fold in range(fold_size):
        train_loop(
            border=border,
            device=device,
            epochs=epochs,
            fold=fold,
            model_name=model_name,
            train=train,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )

    test = pd.read_csv(data_dir + "test.csv")
    predictions = []

    test_dataset = BaseDataset(test, tokenizer, include_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

    for fold in range(hparams.fold):
        model = LightningModel(model_name, border)
        model.to(device)
        model.load_from_checkpoint(output_dir + f"model_{fold:02d}")
        preds = []
        for input_ids, attention_mask in tqdm(test_loader, total=len(test_loader)):
            input_ids: torch.Tensor
            attention_mask: torch.Tensor
            with torch.no_grad():
                y_preds: torch.Tensor = model(input_ids, attention_mask)
            preds.append(y_preds.cpu().numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)
    pd.Series(predictions).to_csv(output_dir + "predictions.csv", index=False)
    predictions1 = np.where(predictions < border, 0, 1)

    # submission
    sub["judgement"] = predictions1
    sub.to_csv(output_dir + "submission.csv", index=False, header=False)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = ArgumentParser()
    parser.add_argument("--fold", default=10, type=int)
    parser.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        type=str,
    )
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--seed", default=472, type=int)
    args = parser.parse_args()

    main(args)
