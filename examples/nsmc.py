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
import numpy as np
from datasets import load_dataset
from functools import partial
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=128, help="")
        parser.add_argument("--max_seq_len", type=int, default=128, help="")
        return parser


def tokenize_function(tokenizer, max_seq_len, example):
    document = example["document"]

    # 문장 시작/끝 토큰 추가
    tokens = [tokenizer.bos_token] + tokenizer.tokenize(document) + [tokenizer.eos_token]
    encoder_input_id = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(encoder_input_id)

    # max_seq_len보다 짧은 경우 패딩
    if len(encoder_input_id) < max_seq_len:
        padding_length = max_seq_len - len(encoder_input_id)
        encoder_input_id += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    else:
        # max_seq_len보다 긴 경우 자르기
        encoder_input_id = encoder_input_id[: max_seq_len - 1] + [
            tokenizer.eos_token_id
        ]
        attention_mask = attention_mask[: max_seq_len]

    return {
        "input_ids": np.array(encoder_input_id, dtype=np.int_),
        "attention_mask": np.array(attention_mask, dtype=float),
        "labels": np.array(example["label"], dtype=np.int_),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtask for KoBART")
    parser.add_argument(
        "--cachedir", type=str, default=os.path.join(os.getcwd(), ".cache")
    )
    parser.add_argument("--subtask", type=str, default="NSMC", help="NSMC")
    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()
    logging.info(args)

    # init model
    model_name = "skt/kobart-base-v1"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.subtask == "NSMC":
        # init data
        dataset = load_dataset("e9t/nsmc", trust_remote_code=True)
    else:
        # add more subtasks
        print("not yet implemented")
        assert False

    tokenized_datasets = dataset.map(
        partial(tokenize_function, tokenizer, args.max_seq_len),
        remove_columns=["document"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir="./results",          # 결과 저장 디렉토리
        learning_rate=5e-5,
        per_device_train_batch_size=64,  # 훈련 배치 크기
        per_device_eval_batch_size=64,   # 평가 배치 크기
        num_train_epochs=5,              # 훈련 에폭 수
        weight_decay=0.01,               # 가중치 감소
        eval_strategy="epoch",     # 에폭마다 평가 수행
        save_strategy="epoch",          # 에폭마다 모델 저장
        load_best_model_at_end=True,     # 훈련 종료 시 가장 좋은 모델 로드
        metric_for_best_model="accuracy", # 가장 좋은 모델을 선택하기 위한 메트릭
        push_to_hub=False,               # Hugging Face Hub에 모델 업로드 여부
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0], axis=1)
        return {"accuracy": np.mean(predictions == labels)}

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # 훈련 시작
    trainer.train()
