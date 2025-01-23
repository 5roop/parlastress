try:
    indata = snakemake.input[0]
    out = snakemake.output[0]
    provenance = snakemake.wildcards.test
    epoch = snakemake.wildcards.epoch
except:
    indata = "brisi.jsonl"
    out = "brisi.png"
    provenance = "IDK"
    epoch = "XX"

import polars as pl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def find_syllable_true(row):
    for i, ii in enumerate(row["nuclei"]):
        if ii == row["true_char_idx"]:
            return i
    raise AttributeError(f"No matches for {row['nuclei']} and {row['true_char_idx']}")


def find_syllable_pred(row):
    for i, ii in enumerate(row["nuclei"]):
        if ii == row["pred_char_idx"]:
            return i
    raise AttributeError(f"No matches for {row['nuclei']} and {row['pred_char_idx']}")


df = pl.read_ndjson(indata).drop_nulls()
df = df.with_columns(
    pl.struct(["true_char_idx", "nuclei"])
    .map_elements(find_syllable_true)
    .alias("true_syl_idx"),
    pl.struct(["pred_char_idx", "nuclei"])
    .map_elements(find_syllable_pred)
    .alias("pred_syl_idx"),
)

y_true = df["true_syl_idx"]
y_pred = df["pred_syl_idx"]
M = max(y_true.max(), y_pred.max())
cm = confusion_matrix(y_true, y_pred)


sns.heatmap(
    cm,
    annot=True,
    cmap="Oranges",
    cbar=False,
    fmt="d",
    xticklabels=[i for i in range(M + 1)],
    yticklabels=[i for i in range(M + 1)],
)
plt.xlabel("Predicted syllable index")
plt.ylabel("True syllable index")
plt.title(f"{provenance} at {epoch} epochs")
plt.savefig(out)
2 + 2
