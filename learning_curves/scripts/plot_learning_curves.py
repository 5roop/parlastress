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
    "PS-RS": "ParlaStress-RS",
    "MP": "MićiPrinc-CKM",
    "SLO": "Artur-SL",
}

df = df.with_columns(
    pl.col("provenance").map_elements(
        lambda s: dataset_renamer.get(s), return_dtype=pl.String
    )
)
fig, axes = plt.subplots(ncols=4, figsize=(13, 4))

for ax, provenance in zip(axes, dataset_renamer.values()):
    l = sns.lineplot(
        df.filter(
            pl.col("steps").eq(1200) & pl.col("provenance").eq(provenance)
        ).to_pandas(),
        x="num_instances",
        y="event_acc",
        # kind="line",
        errorbar="sd",
        ax=ax,
    )
    subset = df.filter(pl.col("steps").eq(6540) & pl.col("provenance").eq(provenance))
    # print(subset.shape)
    mean = subset["event_acc"].mean()
    std = subset["event_acc"].std()

    x = [100, 1000]
    # m = ax.axhline(mean, color="k", linestyle="-.", label="mean", zorder=-100)
    s = ax.axhline(mean + std, color="k", linestyle="--", zorder=-10000, alpha=0.5)
    ax.axhline(mean - std, color="k", linestyle="--", zorder=-10000, alpha=0.5)
    y1 = [mean - std] * 2
    y2 = [mean + std] * 2
    # f = ax.fill_between(x, y1, y2, color="k", alpha=0.1, zorder=5)
    ax.set_ylim((0.8, 1))
    ax.set_title(provenance)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Number of train instances\n\n")
    # ax.get_legend().remove()

l.set_label("mean+/-std")
fig.legend(
    [l.get_children()[0], s],
    [
        r"1200 steps (mean $\pm$ st. dev.)",
        r"all train instances, 6540 steps (mean $\pm$ st. dev.)",
    ],
    ncols=3,
    loc="lower center",
)
plt.tight_layout(w_pad=-0.5)
plt.savefig(pdf_output)
2 + 2
