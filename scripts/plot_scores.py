from pathlib import Path

try:
    infiles = snakemake.input
    outfile = snakemake.output[0]
except NameError:
    infiles = list(Path(".").glob("model*/checkpoint*stats.jsonl"))
    outfile = "brisi.png"
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

df = (
    pl.concat([pl.scan_ndjson(i) for i in infiles])
    .collect()
    .with_columns(
        pl.col("checkpoint").str.split("_").list.get(2).alias("LR"),
        pl.col("checkpoint")
        .str.split("_")
        .list.get(4)
        .str.split("/")
        .list.first()
        .cast(pl.Int32)
        .alias("GAS"),
    )
    .with_columns(
        (
            pl.col("checkpoint")
            .str.split("_")
            .list.get(4)
            .str.split("-")
            .list.last()
            .cast(pl.Int32)
            * 20
            / 6450
            * pl.col("GAS")
        ).alias("epoch"),
    )
)
fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(5, 10))

sns.scatterplot(df.to_pandas(), x="epoch", y="frame_F1", hue="LR", style="GAS", ax=ax1)
ax1.set_ylim((0.8, None))
sns.scatterplot(df.to_pandas(), x="epoch", y="event_F1", hue="LR", style="GAS", ax=ax2)
ax2.set_ylim((0.8, None))
plt.tight_layout()
plt.savefig(outfile)

2 + 2
