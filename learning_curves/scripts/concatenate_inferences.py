try:
    infiles = snakemake.input
    outfile = snakemake.output[0]
except NameError:
    infiles = [
        "models/model_3000_0/checkpoint-150_predictions_MP.jsonl",
        "models/model_3000_0/checkpoint-150_predictions_PS_HR.jsonl",
        "models/model_3000_0/checkpoint-150_predictions_SLO.jsonl",
    ]
    outfile = "brisi.jsonl"
import polars as pl
import pandas as pd

dfs = []
for i in infiles:
    df = pd.read_json(i, lines=True)
    dfs.append(df)


df = pd.concat(
    dfs,
)


df.to_json(outfile, orient="records", lines=True, index=False)
