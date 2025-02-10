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
    "parlastress/prim_stress/RS_Mirna/stress_words_wav2vec2_20_epoch.jsonl",
    "parlastress/prim_stress/PS_Mirna/stress_words_wav2vec2_20_epoch.jsonl",
    "parlastress/prim_stress/MP/stress_words_wav2vec2_20_epoch.jsonl",
    "parlastress/prim_stress/SLO/stress_words_wav2vec2_20_epoch.jsonl",
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

    print(
        f"{l} & {mean_s * 100:2.1f} & [{lo_s * 100:2.1f},{hi_s * 100:0.1f}]  & {mean_t * 100:2.1f}& [{lo_t * 100:2.1f},{hi_t * 100:0.1f}] \\\\"
    )
2 + 2

# import statsmodels.stats.api as sms
# import numpy as np, scipy.stats as st


# a = np.array([1, 1, 2, 1, 1, 5, 2, 1, 3, 1, 2, 12, 3])

# print("statsmodels:")
# print(sms.DescrStatsW(a).tconfint_mean())


# print("scipy.stats.t.interval:")
# print(st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a)))


# print("scipy.stats.t.ppf:")


# def mean_confidence_interval(data, confidence=0.95):
#     import statsmodels.stats.api as sms
#     import numpy as np, scipy.stats as st
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), st.sem(a)
#     h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
#     return m, m - h, m + h


# print(mean_confidence_interval(a))


# import polars as pl

# df = pl.read_ndjson("data/indices_20_epoch.jsonl")
