import polars as pl
from pathlib import Path
import os
from ripostiglio import SAMPLE_RATE, events_to_frames, FRAME_RATE
from datasets import Audio, Dataset

import transformers
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
from transformers import Trainer as Trainer, TrainingArguments as TrainingArguments
from itertools import product
import torch

TARGET = "primstress"
device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

df = (
    pl.read_ndjson("data_PS-HR.jsonl")
    .filter(pl.col("split_speaker").eq("train"))
    .with_columns(pl.col("segment_name").alias("audio"))
)
MAX_DURATION_S = (df["time_e"] - df["time_s"]).max()
df = df.select(["audio", "label"])


ds = Dataset.from_pandas(df.to_pandas()).cast_column(
    "audio", Audio(SAMPLE_RATE, mono=True)
)


feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/w2v-bert-2.0",
)


def preprocess_function(examples):
    inputs = feature_extractor(
        [x["array"] for x in examples["audio"]],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = feature_extractor.pad(
        inputs,
        return_tensors="pt",
        padding="max_length",  # pad to max_length, not just to the longest sequence
        max_length=int(FRAME_RATE * MAX_DURATION_S),
        truncation=True,
    )
    inputs = inputs.convert_to_tensors(tensor_type="pt").to(device)
    labels = []
    for label, tensor in zip(examples["label"], inputs["input_features"]):
        N = tensor.shape[0]
        if N < len(label):
            labels.append(label[:N])
        elif N == len(label):
            labels.append(label)
        else:
            n = N - len(label)
            labels.append([*label, *[[1, 0] for i in range(n)]])
        assert len(labels[-1]) == N
    inputs["label"] = labels

    return inputs


ds = ds.map(
    preprocess_function,
    batched=True,
    batch_size=10,
    remove_columns="audio",
    desc="Extracting features for train",
)


LRs = [
    # "3e-5",
    "1e-5",
    # "8e-6",
    # "5e-5",
]
EPs = [
    "20",
    # "10",
]
gases = [
    1,
    # 4
]

for LR, EP, gas in product(LRs, EPs, gases):
    model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(
        "facebook/w2v-bert-2.0", num_labels=2
    ).cuda()
    print(
        f"Model will be saved to model_{TARGET}_{LR}_{EP}_{gas}",
    )
    D = 1
    training_args = TrainingArguments(
        output_dir=f"model_{TARGET}_{LR}_{EP}_{gas}",
        overwrite_output_dir=True,
        learning_rate=float(LR),
        per_device_train_batch_size=32 // D,
        gradient_accumulation_steps=gas * D,
        num_train_epochs=int(EP),
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=feature_extractor,
    )
    trainer.train()
2 + 2
