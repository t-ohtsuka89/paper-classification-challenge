import math
import os
import random
import time
import warnings
from argparse import ArgumentParser
from logging import Logger

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from dataset import BaseDataset
from model import BaseModel

warnings.filterwarnings("ignore")


def init_logger(log_dir: str, filename: str = "train.log") -> Logger:
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=os.path.join(log_dir, filename))
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed: int = 42):
    # python の組み込み関数の seed を固定
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy の seed を固定
    np.random.seed(seed)
    # torch の seed を固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定論的アルゴリズムを使用する
    torch.backends.cudnn.deterministic = True  # type: ignore


def get_kfold_data(data_frame: pd.DataFrame, fold_num: int, random_state: int) -> pd.DataFrame:
    sfk = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=random_state)
    for n, (_, val_index) in enumerate(sfk.split(data_frame, data_frame["judgement"])):
        data_frame.loc[val_index, "fold"] = int(n)
    data_frame["fold"] = data_frame["fold"].astype(np.uint8)
    return data_frame


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s) -> str:
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent) -> str:
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
):
    start = time.time()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids: torch.Tensor = input_ids.to(device)
        attention_mask: torch.Tensor = attention_mask.to(device)
        labels: torch.Tensor = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(input_ids, attention_mask)

        loss: torch.Tensor = criterion(y_preds, labels)

        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()

        optimizer.step()

        if step % 100 == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    return losses.avg


def valid_fn(valid_loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device):
    start = time.time()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []

    for step, (input_ids, attention_mask, labels) in enumerate(valid_loader):
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        labels: torch.Tensor

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds: torch.Tensor = model(input_ids, attention_mask)

        loss: torch.Tensor = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record score
        preds.append(y_preds.cpu().numpy())

        if step % 100 == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference(
    data_frame: pd.DataFrame,
    device: torch.device,
    fold_size: int,
    models_dir: str,
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    logger: Logger,
):
    predictions = []

    dataset = BaseDataset(data_frame, tokenizer, include_labels=False)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True)

    for fold in range(fold_size):
        logger.info(f"========== model: bert-base-uncased fold: {fold} inference ==========")
        model = BaseModel(model_name)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(models_dir, f"bert-base-uncased_fold{fold}_best.pth"))["model"])
        model.eval()
        preds = []
        for input_ids, attention_mask in tqdm(data_loader, total=len(data_loader)):
            input_ids: torch.Tensor = input_ids.to(device)
            attention_mask: torch.Tensor = attention_mask.to(device)
            with torch.no_grad():
                y_preds: torch.Tensor = model(input_ids, attention_mask)
            preds.append(y_preds.cpu().numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)

    return predictions


def train_loop(
    train: pd.DataFrame,
    fold: int,
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    border: float,
    device: torch.device,
    output_dir: str,
    logger: Logger,
):

    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # Data Loader
    # ====================================================
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

    # ====================================================
    # Model
    # ====================================================
    model = BaseModel(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    criterion = nn.BCELoss()

    # ====================================================
    # Loop
    # ====================================================
    best_score = -1

    for epoch in range(5):
        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds["judgement"].to_numpy()

        # scoring
        score = fbeta_score(valid_labels, np.where(preds < border, 0, 1), beta=7.0)

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch+1} - Score: {score}")

        if score > best_score:
            best_score = score
            logger.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                os.path.join(output_dir, f"bert-base-uncased_fold{fold}_best.pth"),
            )

    check_point = torch.load(os.path.join(output_dir, f"bert-base-uncased_fold{fold}_best.pth"))

    valid_folds["preds"] = check_point["preds"]

    return valid_folds


def get_result(result_df: pd.DataFrame, border: float, logger: Logger):
    preds = result_df["preds"].to_numpy()
    labels = result_df["judgement"].to_numpy()
    score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
    logger.info(f"Score: {score:<.5f}")


def main(hparams):
    data_dir = hparams.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_size: int = hparams.fold
    model_name: str = hparams.model_name
    output_dir: str = hparams.output_dir
    seed = 472
    log_dir = os.path.join(output_dir, "log")

    # fix seed
    seed_torch(seed)

    logger = init_logger(log_dir)

    # load data
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "sample_submit.csv"), header=None)
    sub.columns = ["id", "judgement"]

    train["judgement"][2488] = 0
    train["judgement"][7708] = 0
    border = len(train[train["judgement"] == 1]) / len(train["judgement"])
    train = get_kfold_data(train, fold_size, seed)

    # model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training
    oof_df = pd.DataFrame()
    for fold in range(fold_size):
        _oof_df = train_loop(
            border=border,
            device=device,
            fold=fold,
            logger=logger,
            model_name=model_name,
            output_dir=output_dir,
            train=train,
            tokenizer=tokenizer,
        )
        oof_df = pd.concat([oof_df, _oof_df])
        logger.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df, border=border, logger=logger)

    # CV result
    logger.info(f"========== CV ==========")
    get_result(oof_df, border=border, logger=logger)

    # Save OOF result
    oof_df.to_csv(output_dir + "oof_df.csv", index=False)
    # Inference
    predictions = inference(
        data_frame=test,
        device=device,
        fold_size=fold_size,
        logger=logger,
        model_name=model_name,
        models_dir=output_dir,
        tokenizer=tokenizer,
    )
    pd.Series(predictions).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    predictions1 = np.where(predictions < border, 0, 1)

    # submission
    sub["judgement"] = predictions1
    sub.to_csv(os.path.join(output_dir, "submission.csv"), index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fold", default=10, type=int)
    parser.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        type=str,
    )
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()

    main(args)
