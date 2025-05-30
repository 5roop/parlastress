# Import the main function
from confidence_intervals import evaluate_with_conf_int

# Define the metric of interest (could be a custom method)
from sklearn.metrics import accuracy_score

import numpy as np

svmfiles = [
    "parlastress/prim_stress/PS_Mirna/feat_PS_svm_results.jsonl",
    "parlastress/prim_stress/RS_Mirna/feat_RS_svm_results.jsonl",
    "parlastress/prim_stress/MP/feat_MP_svm_results.jsonl",
    "parlastress/prim_stress/SLO/feat_SLO_svm_results.jsonl",
]
transfiles = [
    "model_primstress_1e-5_20_1/checkpoint-6540_words_PS-HR.jsonl",
    "model_primstress_1e-5_20_1/checkpoint-6540_words_PS-RS.jsonl",
    "model_primstress_1e-5_20_1/checkpoint-6540_words_MP.jsonl",
    "model_primstress_1e-5_20_1/checkpoint-6540_words_SLO.jsonl",
]
labels = ["ParlaStress-HR", "ParlaStress-SR", "MićiPrinc-CKM", "Artur-SLO"]
import polars as pl

for s, t, l in zip(svmfiles, transfiles, labels):
    s = pl.read_ndjson(s)
    t = pl.read_ndjson(t)
    mean_s, (lo_s, hi_s) = evaluate_with_conf_int(
        s["pred_char_idx"], accuracy_score, s["true_char_idx"]
    )
    t = t.with_columns(pl.col("pred_char_idx").fill_null(42))
    mean_t, (lo_t, hi_t) = evaluate_with_conf_int(
        t["pred_char_idx"], accuracy_score, t["true_char_idx"]
    )
    print(t["pred_char_idx"])
    print(
        f"{l} & {mean_s * 100:2.1f} & [{lo_s * 100:2.1f},{hi_s * 100:0.1f}]  & {mean_t * 100:2.1f}& [{lo_t * 100:2.1f},{hi_t * 100:0.1f}] \\\\"
    )
2 + 2
