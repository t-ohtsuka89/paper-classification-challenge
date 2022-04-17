import os
from argparse import ArgumentParser

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from datamodule import MyDataModule
from model_pl import LightningModel
from utils.seed import seed_everything


def save_k_fold(fold_size: int):
    data_dir = "data"
    traincsv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    traincsv["kfold"] = -1
    stratifier = StratifiedKFold(n_splits=fold_size, shuffle=True)

    for fold, (_, val_index) in enumerate(stratifier.split(traincsv, traincsv["judgement"])):
        traincsv.loc[val_index, "kfold"] = fold

    traincsv.to_csv(os.path.join(data_dir, "train_folds.csv"), index=False)


def main(hparams):
    seed_everything()
    save_k_fold(hparams.fold)
    for fold in range(hparams.fold):
        checkpoint_callback = ModelCheckpoint(
            dirpath="models",
            filename=f"model_{fold:02d}",
            monitor="val_fbeta-score",
            verbose=False,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="max",
        )
        model = LightningModel(hparams.model_name, 0.2)
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
        dm = MyDataModule(fold=fold, tokenizer=tokenizer)
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=hparams.max_epochs,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument("--fold", default=10, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    args = parser.parse_args()

    main(args)
