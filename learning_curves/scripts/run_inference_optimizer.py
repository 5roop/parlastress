try:
    model_name = snakemake.input.checkpoint
    outfile = snakemake.output[0]
    datafiles = snakemake.input.datafiles
except NameError:
    model_name = "../model_primstress_3e-5_20_1/checkpoint-6540"
    outfile = model_name + "_predictions.jsonl"
    datafiles = [
        "../data_MP.jsonl",
        "../data_PS-HR.jsonl",
        "../data_SLO.jsonl",
        "../data_PS-RS.jsonl",
    ] * 5


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


device = torch.device("cuda")

df = (
    pl.concat(
        [
            pl.read_ndjson(i, ignore_errors=True, infer_schema_length=None).select(
                """audio label time_s
                time_e split_speaker
                stress segment_name provenance""".split()
            )
            for i in datafiles
        ],
        how="diagonal_relaxed",
    )
    .filter(pl.col("split_speaker").eq("test"))
    .with_columns(pl.col("segment_name").alias("audio"))
)
ds = Dataset.from_pandas(df.to_pandas()).cast_column(
    "audio", Audio(SAMPLE_RATE, mono=True)
)
# df = df.select(["audio", "label"])
try:
    first_path = df["audio"][0]
    assert Path(first_path).exists()
except AssertionError:
    df = df.with_columns(("../" + pl.col("audio")).alias("audio"))
feature_extractor = AutoFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path=model_name
)
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(model_name).to(device)


import numpy as np
import time


def evaluator(chunks):
    import numpy as np

    with torch.no_grad():
        # before = time.time()
        inputs = feature_extractor(
            [i["array"] for i in chunks["audio"]],
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
        )
        # print(f"Extracted features in {time.time() - before:0.5f}")
        # before = time.time()
        inputs = inputs.to(device)
        # print(f"Moved features to GPU in {time.time() - before:0.5f}")
        # before = time.time()
        logits = model(**inputs).logits
        # print(f"Evaluated features with the model in {time.time() - before:0.5f}")
        # before = time.time()
    logits = logits.cpu()
    # print(f"Moved logits to CPU in {time.time() - before:0.5f}")
    # before = time.time()
    y_pred_raw = np.array(logits)
    y_pred = y_pred_raw.argmax(axis=-1)
    # print(f"Did postprocessing in {time.time() - before:0.5f}")
    # before = time.time()
    return {"y_pred": y_pred.tolist(), "y_pred_raw": y_pred_raw}


bs = np.unique(np.logspace(0, 3.7, 300, dtype=np.uint16))
times = []
for b in bs:
    print(b)
    ds = Dataset.from_pandas(df.to_pandas()).cast_column(
        "audio", Audio(SAMPLE_RATE, mono=True)
    )
    try:
        start = time.time()
        ds = ds.map(evaluator, batched=True, batch_size=b)
        end = time.time()
        elapsed = end - start
        times.append(elapsed / ds.shape[0])
        import matplotlib.pyplot as plt

        plt.scatter(bs.tolist()[: len(times)], times)
        plt.xlabel("Batch Size")
        plt.ylabel("Inference Time per 1 Instance (s)")
        plt.savefig("performance4.pdf")
        plt.show()
    except KeyboardInterrupt:
        raise KeyboardInterrupt()
    except Exception as e:
        print(e)
        times.append(None)


2 + 2
