try:
    models = snakemake.params.models
    model_labels = snakemake.params.model_labels
    intextgrid = snakemake.input.intextgrid
    inwav = snakemake.input.inwav
    outwav = snakemake.output.wav
    outtg = snakemake.output.tg
    outerr = snakemake.output.error_report
except:
    models = ["model_primstress_8e-6_20_1/checkpoint-3924"]
    model_labels = ["event optimal"]
    intextgrid = "data/input/PS_Mirna/stress/06cHfZ9Am-Q_1889.84-1906.6.stress.TextGrid"
    inwav = "data/input/PS_Mirna/wav/06cHfZ9Am-Q_1889.84-1906.6.wav"
    outwav = "brisi.wav"
    outtg = "brisi.TextGrid"
    outerr = "brisi.error"

from pathlib import Path
from praatio import textgrid
import polars as pl
import re


pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)


log = ""
tg = textgrid.openTextgrid(intextgrid, includeEmptyIntervals=True)
for model, tier_name in zip(models, model_labels):
    log += f"Log for {model}\n"
    f = model + "_postprocessedpredictions.jsonl"
    target_file = Path(inwav).with_suffix("").name
    df = pl.read_ndjson(f, ignore_errors=True, infer_schema_length=None).filter(
        pl.col("audio").str.contains(target_file)
    )
    for mode, target_column in zip(
        ["raw", "postprocessed"], ["events_pred", "events_pred_pp"]
    ):
        inferred_stresses = []
        for row in df.iter_rows(named=True):
            offset = float(row["time_s"])
            true = row["stress"]
            ts = float(true["time_s"])
            te = float(true["time_e"])
            pred = row[target_column]
            pred = [[round(i[0] + offset, 2), round(i[1] + offset, 2)] for i in pred]
            if len(pred) > 1:
                log += f"{target_file}: Multiple stresses found: {pred}\n"
            for i in pred:
                if i[0] == i[1]:
                    i[1] = i[1] + 0.01
                if (i[0] < te) and (i[1] > ts):
                    is_FP = False
                    # log += f"{target_file}: TRUE positive predicted: {i}\n"
                else:
                    is_FP = True
                    log += f"{target_file}: False positive predicted: {i}\n"
                if is_FP:
                    inferred_stresses.append((i[0], i[1], "FALSE pred"))
                else:
                    inferred_stresses.append((i[0], i[1], "pred"))
                try:
                    tier = textgrid.IntervalTier(
                        tier_name,
                        inferred_stresses,
                        tg.minTimestamp,
                        tg.maxTimestamp,
                    )
                except Exception as e:
                    log += f"{target_file}: Tier construction failed, got error '{str(e).replace('\n', '\t')}', skipping offending predictions\n"
                    inferred_stresses = inferred_stresses[:-1]
        tier = textgrid.IntervalTier(
            tier_name + " " + mode,
            inferred_stresses,
            tg.minTimestamp,
            tg.maxTimestamp,
        )
        tg.addTier(tier)
tg.save(outtg, format="long_textgrid", includeBlankSpaces=True)

Path(outerr).write_text(log)
from shutil import copy

copy(inwav, outwav)
2 + 2
