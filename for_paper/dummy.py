import polars as pl
from sklearn.dummy import DummyClassifier


def get_index(row):
    stress_start = row["stress"]["time_s"]
    nonstress_starts = [i["time_s"] for i in row["unstress"]]
    all = sorted(nonstress_starts + [stress_start])
    for i, t in enumerate(all):
        if t == stress_start:
            return i + 1


hr = pl.read_ndjson("/cache/peterr/parlastress/data_PS-HR.jsonl").with_columns(
    pl.struct(["stress", "unstress"])
    .map_elements(get_index, return_dtype=int)
    .alias("stress_ind")
)
rs = pl.read_ndjson("/cache/peterr/parlastress/data_PS-RS.jsonl").with_columns(
    pl.struct(["stress", "unstress"])
    .map_elements(get_index, return_dtype=int)
    .alias("stress_ind")
)
sl = pl.read_ndjson("/cache/peterr/parlastress/data_SLO.jsonl").with_columns(
    pl.struct(["stress", "unstress"])
    .map_elements(get_index, return_dtype=int)
    .alias("stress_ind")
)
ckm = pl.read_ndjson("/cache/peterr/parlastress/data_MP.jsonl").with_columns(
    pl.struct(["stress", "unstress"])
    .map_elements(get_index, return_dtype=int)
    .alias("stress_ind")
)


train = hr.filter(pl.col("split_speaker").eq("train"))
y_train = train["stress_ind"].to_list()

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


for lang, df in zip(("HR", "RS", "SL", "CKM"), (hr, rs, sl, ckm)):
    for strategy in ("most_frequent", "stratified"):
        dummy_clf = DummyClassifier(strategy=strategy)
        dummy_clf.fit(y_train, y_train)
        y_true = df.filter(pl.col("split_speaker").eq("test"))["stress_ind"].to_list()
        y_pred = dummy_clf.predict(y_true)
        print(f"{lang=}, {strategy=}, accuracy: ", accuracy_score(y_true, y_pred))


2 + 2
