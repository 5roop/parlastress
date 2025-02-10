try:
    files = snakemake.input
    output = snakemake.output[0]
    pdf_output = snakemake.output[1]
except NameError:
    from pathlib import Path

    files = list(Path("models/").glob("model_*/checkpoint*stats.jsonl"))
    output = "brisi.png"
    # pdf_output = "../paper_images/learning_curves.pdf"
    pdf_output = "brisi.pdf"

import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.concat([pd.read_json(i) for i in files]).dropna()

df = (
    pl.from_pandas(df)
    .with_columns(
        pl.col("checkpoint")
        .str.split("_")
        .list.get(1)
        .cast(pl.UInt16)
        .alias("num_instances"),
        pl.col("checkpoint")
        .str.extract(r"model_\d+_(\d+)", 1)
        .cast(pl.UInt32)
        .alias("steps"),
    )
    .filter(pl.col("mode").eq("pp"))
)

# df = df.pivot("mode", index="mode provenance".split(), values="event_F1 TP FN FP".split())
sns.relplot(
    df.filter(pl.col("steps").ne(6540)).to_pandas(),
    x="num_instances",
    y="event_acc",
    col="provenance",
    # hue="mode",
    kind="line",
    row="steps",
    height=4,
    errorbar="sd",
    # facet_kws={"sharey":False}
)
for i in plt.gcf().axes:
    provenance = i.title._text.split("=")[-1].strip()
    subset = df.filter(pl.col("steps").eq(6540) & pl.col("provenance").eq(provenance))
    # print(subset.shape)
    mean = subset["event_acc"].mean()
    std = subset["event_acc"].std()

    i.axhline(mean, color="k", linestyle="-")
    i.axhline(mean + std, color="k", linestyle="--")
    i.axhline(mean - std, color="k", linestyle="--")


plt.gcf().axes[-1].set_ylim((None, 1))
plt.gcf().savefig(output)


dataset_renamer = {
    "PS-HR": "ParlaStress-HR",
    "PS-RS": "ParlaStress-SR",
    "MP": "MićiPrinc-CKM",
    "SLO": "Artur-SL",
}

df = df.with_columns(
    pl.col("provenance").map_elements(
        lambda s: dataset_renamer.get(s), return_dtype=pl.String
    )
)
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple, HandlerBase
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

F = 13
plt.rcParams.update({"font.size": F})
D = 1.8
fig, axes = plt.subplots(ncols=4, figsize=(D * 17 / 2.54, D * 5 / 2.54))

for ax, provenance in zip(axes, dataset_renamer.values()):
    l = sns.lineplot(
        df.filter(
            pl.col("steps").eq(1200) & pl.col("provenance").eq(provenance)
        ).to_pandas(),
        x="num_instances",
        y="event_acc",
        # kind="line",
        errorbar="sd",  # ("ci", 95),
        ax=ax,
    )
    subset = df.filter(pl.col("steps").eq(6540) & pl.col("provenance").eq(provenance))
    # print(subset.shape)
    mean = subset["event_acc"].mean()
    std = subset["event_acc"].std()

    x = [100, 1000]
    m = ax.axhline(
        mean,
        color="k",
        linestyle="-",
        label="mean",
        zorder=-100,
    )
    s = ax.axhline(mean + std, color="k", linestyle="dotted", zorder=-10000, alpha=0.7)
    ax.axhline(mean - std, color="k", linestyle="dotted", zorder=-10000, alpha=0.7)
    y1 = [mean - std] * 2
    y2 = [mean + std] * 2
    # f = ax.fill_between(x, y1, y2, color="k", alpha=0.1, zorder=5)
    ax.set_ylim((0.8, 1))
    ax.set_title(provenance, fontsize=F)
    ax.set_ylabel("Accuracy", fontsize=F)
    ax.set_xlabel("Number of Train Instances", fontsize=F)
    # ax.get_legend().remove()
    if not ax == axes[0]:
        ax.set_yticklabels([])
        ax.set_ylabel(None)


dummy_patch = mpatches.Patch(color="white", alpha=0)
main_legend = fig.legend(
    [
        dummy_patch,
        l.get_children()[0],
        l.get_children()[1],
        dummy_patch,
        m,
        s,
    ],
    [
        "Limited Train Data:",
        "Mean",
        "St. Dev.",
        "Full Train Data:",
        "Mean",
        "St. Dev.",
    ],
    ncols=6,
    loc="lower center",
    # bbox_to_anchor=(0.5, 0, 0.5, 0.5),
    # mode="expand",
)
fig.subplots_adjust(bottom=0.33, wspace=0.1, right=0.98, left=0.07)
# fig.suptitle("Learning curves")
plt.savefig(pdf_output)


def mean_confidence_interval(data, confidence=0.95):
    import statsmodels.stats.api as sms
    import numpy as np, scipy.stats as st

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
    return {
        "mean": m,
        f"ci_{confidence}_lower": m - h,
        f"ci_{confidence}_higher": m + h,
    }


gb = (
    df.filter(pl.col("steps").is_in([6540]))
    .group_by(
        [
            "provenance",
            "num_instances",
            "steps",
        ]
    )
    .agg(
        pl.col("event_acc")
        .map_elements(mean_confidence_interval, return_dtype=pl.Struct)
        .alias("scores")
    )
    .unnest("scores")
).sort(["provenance", "num_instances"])

pl.Config.set_tbl_cols(n=-1)
pl.Config.set_fmt_str_lengths(20)
pl.Config.set_tbl_width_chars(800)
pl.Config.set_tbl_rows(-1)

print(gb)
# print(gb)
2 + 2
