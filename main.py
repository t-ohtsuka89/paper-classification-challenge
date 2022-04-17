import math
import os
import random
import time
import warnings

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data/"
OUTPUT_DIR = "./log/"


def init_logger(log_file: str = OUTPUT_DIR + "train.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


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


seed = 472
seed_torch(seed)

train = pd.read_csv(DATA_DIR + "train.csv")
test = pd.read_csv(DATA_DIR + "test.csv")
sub = pd.read_csv(DATA_DIR + "sample_submit.csv", header=None)
sub.columns = ["id", "judgement"]

train["judgement"][2488] = 0
train["judgement"][7708] = 0

# この値を境に、CV値を算出する
border = len(train[train["judgement"] == 1]) / len(train["judgement"])
print(border)

Fold = 10


def get_train_data(train: pd.DataFrame):
    # 交差検証 用の番号を振ります。
    fold = StratifiedKFold(n_splits=Fold, shuffle=True, random_state=seed)
    for n, (_, val_index) in enumerate(fold.split(train, train["judgement"])):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(np.uint8)

    return train


def get_test_data(test):
    return test


train = get_train_data(train)


model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
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
    start = end = time.time()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
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


def valid_fn(valid_loader, model: nn.Module, criterion, device: torch.device):
    start = end = time.time()
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
            y_preds = model(input_ids, attention_mask)

        loss = criterion(y_preds, labels)
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


def inference():
    predictions = []

    test_dataset = BaseDataset(test, tokenizer, include_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)

    for fold in range(Fold):
        LOGGER.info(f"========== model: bert-base-uncased fold: {fold} inference ==========")
        model = BaseModel(model_name)
        model.to(device)
        model.load_state_dict(torch.load(OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth")["model"])
        model.eval()
        preds = []
        for i, (input_ids, attention_mask) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                y_preds: torch.Tensor = model(input_ids, attention_mask)
            preds.append(y_preds.cpu().numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)

    return predictions


def train_loop(train: pd.DataFrame, fold: int):

    LOGGER.info(f"========== fold: {fold} training ==========")

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
    best_loss = np.inf

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
        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score}")

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth",
            )

    check_point = torch.load(OUTPUT_DIR + f"bert-base-uncased_fold{fold}_best.pth")

    valid_folds["preds"] = check_point["preds"]

    return valid_folds


def get_result(result_df: pd.DataFrame):
    preds = result_df["preds"].to_numpy()
    labels = result_df["judgement"].to_numpy()
    score = fbeta_score(labels, np.where(preds < border, 0, 1), beta=7.0)
    LOGGER.info(f"Score: {score:<.5f}")


# Training
oof_df = pd.DataFrame()
for fold in range(Fold):
    _oof_df = train_loop(train, fold)
    oof_df = pd.concat([oof_df, _oof_df])
    LOGGER.info(f"========== fold: {fold} result ==========")
    get_result(_oof_df)

# CV result
LOGGER.info(f"========== CV ==========")
get_result(oof_df)

# Save OOF result
oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)
# Inference
predictions = inference()
pd.Series(predictions).to_csv(OUTPUT_DIR + "predictions.csv", index=False)
predictions1 = np.where(predictions < 0.0262, 0, 1)

# submission
sub["judgement"] = predictions1
sub.to_csv(OUTPUT_DIR + "submission.csv", index=False, header=False)
