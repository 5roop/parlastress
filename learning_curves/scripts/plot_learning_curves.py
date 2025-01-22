try:
    files = snakemake.input
    output = snakemake.output[0]
except NameError:
    from pathlib import Path

    files = list(Path("models/").glob("model_*/checkpoint*stats.jsonl"))
    output = "brisi.png"

import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.concat(
    [
        pd.read_json(
            i,
        )
        for i in files
    ]
)

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


2 + 2
