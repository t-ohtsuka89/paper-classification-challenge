import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from dataset import BaseDataset
from model import BaseModel

warnings.filterwarnings("ignore")


def main(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(os.path.join(hparams.data_dir, "test.csv"))
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    test_dataset = BaseDataset(test, tokenizer, include_labels=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)
    predictions = []
    for log_dir in ["log", "log_v1"]:
        for fold in range(hparams.fold):
            model = BaseModel(hparams.model_name)
            model.to(device)
            model.load_state_dict(torch.load(os.path.join(log_dir, f"bert-base-uncased_fold{fold}_best.pth"))["model"])
            model.eval()
            preds = []
            for i, (input_ids, attention_mask) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input_ids: torch.Tensor = input_ids.to(device)
                attention_mask: torch.Tensor = attention_mask.to(device)
                with torch.no_grad():
                    y_preds: torch.Tensor = model(input_ids, attention_mask)
                preds.append(y_preds.cpu().numpy())
            preds = np.concatenate(preds)
            predictions.append(preds)
    predictions = np.mean(predictions, axis=0)

    sub = pd.read_csv(os.path.join(hparams.data_dir, "sample_submit.csv"), header=None)
    sub.columns = ["id", "judgement"]
    pd.Series(predictions).to_csv(os.path.join(hparams.output_dir, "predictions.csv"), index=False)
    predictions1 = np.where(predictions < 0.0262, 0, 1)

    sub["judgement"] = predictions1
    sub.to_csv(os.path.join("outputs", "submission.csv"), index=False, header=False)


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
