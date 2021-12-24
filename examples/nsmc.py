# coding=utf-8
# Modified MIT License

# Software Copyright (c) 2020 SK telecom

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartForSequenceClassification

from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
from kobart import download


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=128, help="")
        parser.add_argument("--max_seq_len", type=int, default=128, help="")
        return parser


class NSMCDataset(Dataset):
    def __init__(self, filepath, max_seq_len=128):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath, sep="\t")
        self.max_seq_len = max_seq_len
        self.tokenizer = get_kobart_tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        document, label = str(record["document"]), int(record["label"])
        tokens = (
            [self.tokenizer.bos_token]
            + self.tokenizer.tokenize(document)
            + [self.tokenizer.eos_token]
        )
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(encoder_input_id)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            encoder_input_id = encoder_input_id[: self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id
            ]
            attention_mask = attention_mask[: self.max_seq_len]
        return {
            "input_ids": np.array(encoder_input_id, dtype=np.int_),
            "attention_mask": np.array(attention_mask, dtype=float),
            "labels": np.array(label, dtype=np.int_),
        }


class NSMCDataModule(pl.LightningDataModule):
    def __init__(self, max_seq_len=128, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        s3_train_file = {
            "url": "s3://skt-lsl-nlp-model/KoBART/datasets/nsmc/ratings_train.txt",
            "chksum": None,
        }
        s3_test_file = {
            "url": "s3://skt-lsl-nlp-model/KoBART/datasets/nsmc/ratings_test.txt",
            "chksum": None,
        }

        os.makedirs(os.path.dirname(args.cachedir), exist_ok=True)
        self.train_file_path, is_cached = download(
            s3_train_file["url"], s3_train_file["chksum"], cachedir=args.cachedir
        )
        self.test_file_path, is_cached = download(
            s3_test_file["url"], s3_test_file["chksum"], cachedir=args.cachedir
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.nsmc_train = NSMCDataset(self.train_file_path, self.max_seq_len)
        self.nsmc_test = NSMCDataset(self.test_file_path, self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        nsmc_train = DataLoader(
            self.nsmc_train, batch_size=self.batch_size, num_workers=5, shuffle=True
        )
        return nsmc_train

    def val_dataloader(self):
        nsmc_val = DataLoader(
            self.nsmc_test, batch_size=self.batch_size, num_workers=5, shuffle=False
        )
        return nsmc_val

    def test_dataloader(self):
        nsmc_test = DataLoader(
            self.nsmc_test, batch_size=self.batch_size, num_workers=5, shuffle=False
        )
        return nsmc_test


class Classification(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Classification, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="batch size for training (default: 96)",
        )

        parser.add_argument(
            "--lr", type=float, default=5e-5, help="The initial learning rate"
        )

        parser.add_argument(
            "--warmup_ratio", type=float, default=0.1, help="warmup ratio"
        )

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus is not None else 1) * (
            self.hparams.num_nodes if self.hparams.num_nodes is not None else 1
        )
        data_len = len(self.train_dataloader().dataset)
        logging.info(f"number of workers {num_workers}, data length {data_len}")
        num_train_steps = int(
            data_len
            / (
                self.hparams.batch_size
                * num_workers
                * self.hparams.accumulate_grad_batches
            )
            * self.hparams.max_epochs
        )
        logging.info(f"num_train_steps : {num_train_steps}")
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f"num_warmup_steps : {num_warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]


class KoBARTClassification(Classification):
    def __init__(self, hparams, **kwargs):
        super(KoBARTClassification, self).__init__(hparams, **kwargs)
        self.model = BartForSequenceClassification.from_pretrained(
            get_pytorch_kobart_model()
        )
        self.model.train()
        self.metric_acc = pl.metrics.classification.Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = outs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["input_ids"], batch["attention_mask"])
        labels = batch["labels"]
        accuracy = self.metric_acc(
            torch.nn.functional.softmax(pred.logits, dim=1), labels
        )
        self.log("accuracy", accuracy)
        result = {"accuracy": accuracy}
        # Checkpoint model based on validation loss
        return result

    def validation_epoch_end(self, outputs):
        val_acc = torch.stack([i["accuracy"] for i in outputs]).mean()
        self.log("val_acc", val_acc, prog_bar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtask for KoBART")
    parser.add_argument(
        "--cachedir", type=str, default=os.path.join(os.getcwd(), ".cache")
    )
    parser.add_argument("--subtask", type=str, default="NSMC", help="NSMC")
    parser = Classification.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = NSMCDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    if args.default_root_dir is None:
        args.default_root_dir = args.cachedir

    # init model
    model = KoBARTClassification(args)

    if args.subtask == "NSMC":
        # init data
        dm = NSMCDataModule(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            dirpath=args.default_root_dir,
            filename="model_chp/{epoch:02d}-{val_acc:.3f}",
            verbose=True,
            save_last=True,
            mode="max",
            save_top_k=-1,
            prefix=f"{args.subtask}",
        )
    else:
        # add more subtasks
        assert False
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(args.default_root_dir, "tb_logs")
    )
    # train
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback, lr_logger]
    )
    trainer.fit(model, dm)
