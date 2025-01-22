try:
    instances = snakemake.wildcards.instances
    output_dir = snakemake.output[0]
    serial = snakemake.wildcards.serial
    data = snakemake.input.data
    steps = snakemake.wildcards.steps
except NameError:
    instances = "4800"
    output_dir = "brisi"
    serial = "0"
    steps = "300"
    data = "../data_PS-HR.jsonl"

instances = int(instances)
serial = int(serial)
steps = int(steps)
output_dir = f"models/model_{instances}_{steps}_{serial}/"
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

LR = 1e-5
STEPS = int(steps)

device = torch.device("cuda")
df = (
    pl.read_ndjson(data)
    .filter(pl.col("split_speaker").eq("train"))
    .with_columns(("../" + pl.col("segment_name")).alias("audio"))
)

# Selecting by sentences:
sentences = df["audio_wav"].unique().shuffle()
selection = []
i = 0
while len(selection) <= instances:
    sentence = sentences[i]
    samples = df.filter(pl.col("audio_wav").eq(sentence))["audio"].to_list()
    selection.extend(samples)
    selection = list(set(selection))
    i += 1
selection = selection[:instances]
# This is where the selection is applied:
df = df.filter(pl.col("audio").is_in(selection))
# Assure that we really have the desired number of samples
print(f"{instances=}, {df.shape=}")
MAX_DURATION_S = (df["time_e"] - df["time_s"]).max()
df = df.select(["audio", "label"])

ds = Dataset.from_pandas(df.to_pandas()).cast_column(
    "audio", Audio(SAMPLE_RATE, mono=True)
)
print(f"{len(ds['audio'])}")

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

model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(
    "facebook/w2v-bert-2.0", num_labels=2
).cuda()

BATCH_SIZE = 32
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    learning_rate=float(LR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    save_strategy="steps",
    max_steps=STEPS,
    save_steps=STEPS,
    logging_steps=10,
    logging_strategy="steps",
    save_total_limit=1,
    no_cuda=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=feature_extractor,
)
trainer.train()
