import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, include_labels=True):
        self.df = df
        self.include_labels = include_labels

        df["title_abstract"] = df["title"] + " " + df["abstract"].fillna("")
        sentences = df["title_abstract"].tolist()

        max_length = 512
        self.encoded = tokenizer.batch_encode_plus(
            sentences,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        if self.include_labels:
            self.labels = df["judgement"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded["input_ids"][idx])  # type: ignore
        attention_mask = torch.tensor(self.encoded["attention_mask"][idx])  # type: ignore

        if self.include_labels:
            label = torch.tensor(self.labels[idx]).float()
            return input_ids, attention_mask, label

        return input_ids, attention_mask
