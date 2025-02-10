try:
    indata = snakemake.input[0]
    out = snakemake.output[0]
except:
    indata = "brisi.jsonl"
    out = "brisi.png"

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


dataset_renamer = {
    "PS-HR": "ParlaStress-HR",
    "PS-RS": "ParlaStress-SR",
    "MP": "MićiPrinc-CKM",
    "SLO": "Artur-SL",
}

df = pl.read_ndjson(indata, ignore_errors=True, infer_schema_length=None).drop_nulls()
df = df.with_columns(
    pl.struct(["true_char_idx", "nuclei"])
    .map_elements(find_syllable_true)
    .alias("true_syl_idx"),
    pl.struct(["pred_char_idx", "nuclei"])
    .map_elements(find_syllable_pred)
    .alias("pred_syl_idx"),
    pl.col("provenance").map_elements(
        lambda s: dataset_renamer.get(s), return_dtype=pl.String
    ),
)
provenances = df["provenance"].unique().to_list()
provenances = sorted(
    provenances,
    key=lambda s: {
        "ParlaStress-HR": 0,
        "ParlaStress-SR": 1,
        "MićiPrinc-CKM": 2,
        "Artur-SL": 3,
    }.get(s, 55),
)

MAX_SYLLABLE = 5
num_rows = len(provenances)
D = 1
fig, axes = plt.subplots(ncols=num_rows, figsize=(10 * D, 3 * D))
for provenance, ax in zip(provenances, axes):
    subset = df.filter(pl.col("provenance").eq(provenance))
    accuracy = (
        subset["true_char_idx"] == subset["pred_char_idx"]
    ).sum() / subset.shape[0]
    y_true = subset["true_syl_idx"] + 1
    y_pred = subset["pred_syl_idx"] + 1
    cm = confusion_matrix(y_true, y_pred, labels=[i + 1 for i in range(MAX_SYLLABLE)])
    sns.heatmap(
        cm,
        annot=True,
        cmap="Oranges",
        cbar=False,
        fmt="d",
        xticklabels=[f"{i}" for i in range(1, MAX_SYLLABLE + 1)],
        yticklabels=[f"{i}" for i in range(1, MAX_SYLLABLE + 1)],
        ax=ax,
    )
    ax.set_xlabel("Predicted Stressed Syllable")
    ax.set_ylabel("True Stressed Syllable")
    ax.set_aspect("equal")
    ax.set_title(
        provenance
        #  + f"\nAccuracy: {100 * accuracy:0.1f}%"
    )
plt.tight_layout()
# fig.suptitle(str(Path(indata).with_suffix("")))
plt.savefig(out)
2 + 2
