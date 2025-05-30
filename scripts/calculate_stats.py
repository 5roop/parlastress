try:
    infile = snakemake.input[0]
    outfile = snakemake.output[0]
    modes = snakemake.params.modes
    provenances = snakemake.params.provenances
except NameError:
    infile = "models/model_1000_300_1/checkpoint-300_postprocessedpredictions.jsonl"
    outfile = "brisi.jsonl"
    modes = ["raw", "pp"]
    provenances = ["PS-HR", "MP", "SLO", "PS-RS"]


import polars as pl
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from numpy import nan
import json

df = pl.read_ndjson(infile, ignore_errors=True, infer_schema_length=None)


def eval_events(row):
    from pandas import Interval

    trues = [Interval(i[0], i[1]) for i in row["events_true"]]
    try:
        preds = [Interval(i[0], i[1]) for i in row["events_pred"]]
    except:
        preds = [Interval(i[0], i[1]) for i in row["events_pred_pp"]]
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


results = []
for mode in modes:
    for provenance in provenances:
        df = pl.read_ndjson(
            infile, ignore_errors=True, infer_schema_length=None
        ).filter(pl.col("provenance").eq(provenance))
        event_pred_col = "events_pred_pp" if mode == "pp" else "events_pred"
        df = df.with_columns(
            pl.struct(f"events_true {event_pred_col}".split())
            .map_elements(eval_events)
            .alias("event_stats"),
            pl.struct("y_true y_pred".split())
            .map_elements(eval_frames)
            .alias("frame_stats"),
        ).unnest(["event_stats", "frame_stats"])

        # Events:
        TP = df["event_TP"].sum()
        FN = df["event_FN"].sum()
        FP = df["event_FP"].sum()
        try:
            ep = TP / (TP + FP)
        except ZeroDivisionError:
            ep = nan
        try:
            er = TP / (TP + FN)
        except ZeroDivisionError:
            er = nan
        event_F1 = 2 * ep * er / (ep + er)

        # # frames:
        # TP = df["frame_TP"].sum()
        # FN = df["frame_FN"].sum()
        # FP = df["frame_FP"].sum()
        # try:
        #     fp = TP / (TP + FP)
        # except ZeroDivisionError:
        #     fp = nan
        # try:
        #     fr = TP / (TP + FN)
        # except ZeroDivisionError:
        #     fr = nan
        # frame_F1 = 2 * fp * fr / (fp + fr)

        r = {
            "checkpoint": infile,
            "event_precision": ep,
            "event_recall": er,
            "event_F1": event_F1,
            "event_acc": ep,  # TP / (TP + FP),
            "TP": TP,
            "FN": FN,
            "FP": FP,
            # "frame_precision": fp,
            # "frame_recall": fr,
            # "frame_F1": frame_F1,
            "provenance": provenance,
            "mode": mode,
        }
        results.append(r)
assert len(results) == len(modes) * len(provenances)

Path(outfile).write_text(json.dumps(results, indent=4))
2 + 2
