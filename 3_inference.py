import polars as pl
from pathlib import Path
import os
import numpy as np
from ripostiglio import SAMPLE_RATE, events_to_frames, FRAME_RATE, frames_to_events
from datasets import Audio, Dataset

import transformers
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
from itertools import product
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda")

df = (
    pl.read_ndjson("data.jsonl")
    .filter(pl.col("split_speaker").eq("test"))
    .with_columns(pl.col("segment_name").alias("audio"))
)
# df = df.select(["audio", "label"])

model_name = "model_primstress_3e-5_20_1/checkpoint-6540"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path=model_name
)
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(model_name).to(device)


ds = Dataset.from_pandas(df.to_pandas()).cast_column(
    "audio", Audio(SAMPLE_RATE, mono=True)
)


def evaluator(chunks):
    import numpy as np

    with torch.no_grad():
        inputs = feature_extractor(
            [i["array"] for i in chunks["audio"]],
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        logits = model(**inputs).logits
    y_pred_raw = np.array(logits.cpu())
    y_pred = y_pred_raw.argmax(axis=-1)
    return {"y_pred": y_pred.tolist(), "y_pred_raw": y_pred_raw}


ds = ds.map(evaluator, batched=True, batch_size=10)
y_pred = pl.Series("y_pred", ds["y_pred"])

df = df.insert_column(-1, y_pred)
df = (
    df.with_columns(
        pl.col("label")
        .map_elements(lambda l: np.array(l.to_list()).argmax(axis=-1).tolist())
        .alias("y_true"),
    )
    .with_columns(
        pl.col("y_true").list.len().alias("m"),
        pl.col("y_pred").list.len().alias("n"),
    )
    .with_columns(
        pl.struct(["m", "n"])
        .map_elements(lambda row: min(row["m"], row["n"]))
        .alias("min")
    )
    .with_columns(
        pl.struct("y_pred min".split())
        .map_elements(lambda row: row["y_pred"][: row["min"]])
        .alias("y_pred"),
        pl.struct("y_true min".split())
        .map_elements(lambda row: row["y_true"][: row["min"]])
        .alias("y_true"),
    )
    .with_columns(
        pl.col("y_true").list.len().alias("m"),
        pl.col("y_pred").list.len().alias("n"),
    )
).with_columns(
    pl.col("y_pred").map_elements(frames_to_events).alias("events_pred"),
    pl.struct("stress time_s".split())
    .map_elements(
        lambda d: [
            [d["stress"]["time_s"] - d["time_s"], d["stress"]["time_e"] - d["time_s"]]
        ]
    )
    .alias("events_true"),
)

df.select("audio time_s time_e stress y_pred y_true segment_name  events_true events_pred".split()).write_ndjson(
    model_name + "_predictions.jsonl"
)
2 + 2
