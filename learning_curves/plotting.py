import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

files = list(Path("models").glob("model*/*stats.jsonl"))
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
    .filter(pl.col("provenance").is_in(["SLO", "MP"]))
    .filter(pl.col("steps") > 150)
    .filter(pl.col("num_instances").is_between(100, 1000))
).select(
    [
        "event_acc",
        "steps",
        "num_instances",
        "provenance",
    ]
)
gdf = (
    df.with_columns(pl.col("steps").cast(pl.String))
    .group_by(["num_instances", "steps", "provenance"])
    .agg(
        pl.col("event_acc").mean().alias("event_acc"),
        pl.col("event_acc").std().alias("event_acc_std"),
    )
)
sns.relplot(
    gdf.to_pandas(),
    x="num_instances",
    hue="steps",
    y="event_acc",
    col="provenance",
    kind="line",
    errorbar="sd",
    # err_style="bars",
)

plt.savefig("brisi.png")

print(
    gdf.group_by(["steps", "provenance"])
    .agg(
        pl.col("event_acc").mean().alias("event_acc"),
        pl.col("event_acc").std().alias("event_acc_std"),
    )
    .with_columns(pl.col("steps").cast(pl.Int32))
    .sort(["provenance", "steps"])
)

2 + 2
