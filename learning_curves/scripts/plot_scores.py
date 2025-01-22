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
    pl.concat([pl.read_json(i) for i in infiles])
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


g = sns.relplot(
    df.to_pandas(),
    x="epoch",
    y="event_F1",
    hue="LR",
    style="GAS",
    row="provenance",
    col="mode",
)
for ax in g.axes.reshape(-1):
    ax.set_ylim((0.8, None))
plt.gcf().savefig(outfile)

ss = df.filter(pl.col("provenance").eq("PS-HR") & pl.col("mode").eq("pp"))
max_f1 = ss["event_F1"].max()
best_checkpoints = ss.sort(pl.col("event_F1"), descending=True).head(5)
print(best_checkpoints)

presumed_best_checkpoint = best_checkpoints["checkpoint"][0]
print("Calculating stats for presumed best checkpoint:", presumed_best_checkpoint)

ndf = df.filter(pl.col("checkpoint").eq(presumed_best_checkpoint))


with pl.Config(set_tbl_cols=15):
    print(ndf.pivot("mode", index="provenance", values="event_F1 TP FN FP".split()))


2 + 2
