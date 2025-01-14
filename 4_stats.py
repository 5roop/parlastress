import polars as pl
import numpy as np
from pathlib import Path
from numpy import nan
import json

f = "model_primstress_3e-5_20_1/checkpoint-6540_predictions.jsonl"

df = pl.read_ndjson(f)


def eval_events(row):
    from pandas import Interval

    trues = [Interval(i[0], i[1]) for i in row["events_true"]]
    preds = [Interval(i[0], i[1]) for i in row["events_pred"]]
    from ripostiglio import events_overlap, extract

    TP, FN, FP = extract(trues, preds)
    return dict(event_TP=len(TP), event_FN=len(FN), event_FP=len(FP))


def eval_frames(row):
    from sklearn.metrics import confusion_matrix

    trues = row["y_true"]
    preds = row["y_pred"]
    cm = confusion_matrix(trues, preds)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    return dict(frame_TP=TP, frame_FN=FN, frame_FP=FP)


df = df.with_columns(
    pl.struct("events_true events_pred".split())
    .map_elements(eval_events)
    .alias("event_stats"),
    pl.struct("y_true y_pred".split()).map_elements(eval_frames).alias("frame_stats"),
).unnest(["event_stats", "frame_stats"])


# Events:
TP = df["event_TP"].sum()
FN = df["event_FN"].sum()
FP = df["event_FP"].sum()
try:
    ep = TP / (TP+FP)
except ZeroDivisionError:
    ep=nan
try:
    er = TP / (TP + FN)
except ZeroDivisionError:
    er=nan
event_F1 = 2 * ep * er / (ep + er )


# frames:
TP = df["frame_TP"].sum()
FN = df["frame_FN"].sum()
FP = df["frame_FP"].sum()
try:
    fp = TP / (TP+FP)
except ZeroDivisionError:
    fp=nan
try:
    fr = TP / (TP + FN)
except ZeroDivisionError:
    fr=nan
frame_F1 = 2 * fp * fr / (fp + fr )

r = {
    "checkpoint": f,
    "event_precision": ep,
    "event_recall": er,
    "event_F1": event_F1,
    "frame_precision": fp,
    "frame_recall": fr,
    "frame_F1": frame_F1
}
Path(f.replace("_predictions", "_stats")).write_text(json.dumps(r))
2 + 2
