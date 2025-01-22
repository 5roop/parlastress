try:
    infile = snakemake.input.predictions
    outfile = snakemake.output[0]
    datafiles = snakemake.input.datafiles
except NameError:
    infile = "models/model_2400_0/checkpoint-150_predictions.jsonl"
    outfile = "models/model_2400_0_checkpoint-150_postprocessedpredictions.jsonl"
    datafiles = ["../data_MP.jsonl", "../data_PS-HR.jsonl", "../data_SLO.jsonl"]

import polars as pl

pl.Config(set_fmt_str_lengths=None)
import numpy as np

df = pl.concat(
    [
        pl.read_ndjson(i).select(
            """segment_name
                               stress unstress""".split()
        )
        for i in datafiles
    ],
    how="vertical_relaxed",
)

preds = pl.read_ndjson(infile, ignore_errors=True)


# Fix dual stress predictions: take longest:
def fix_dual(events: list[list[float]]) -> list[list[float]]:
    try:
        events = events.to_list()
        if (L := len(events)) <= 1:
            return events
        else:
            # print(events, type(events))
            durations = [i[1] - i[0] for i in events]
            longest = np.argmax(durations)
            # print(events[longest])
            return [events[longest]]
    except Exception as e:
        return None


preds = preds.with_columns(
    pl.col("events_pred").map_elements(fix_dual).alias("events_pred_pp")
).drop_nulls(subset="events_pred_pp")

preds = preds.join(
    df.select(["segment_name", "stress", "unstress"]).unique(),
    how="left",
    on="segment_name",
)


# Fix incorrect stress:d
def fix_incorrect(row):
    try:
        predicted = row["events_pred_pp"]
        if len(predicted) == 0:
            return predicted
        # Check if prediction coincides with stress carier:
        carriers = [row["stress"], *row["unstress"]]
        stress_s, stress_e = row["stress"]["time_s"], row["stress"]["time_e"]
        time_e = row["time_e"]
        time_s = row["time_s"]
        # events_pred_pp are in word timeframe, all the rest is in segment timeframe
        #  We'll work in segment TF and backconvert later
        predicted_s, predicted_e = predicted[0][0] + time_s, predicted[0][1] + time_s

        # # Do we have a hit?
        # if (predicted_s > stress_e) and (predicted_e > stress_s):
        #     # We doo
        #     return predicted
        # # We dont: let's find the nearest potential stress carrier

        # Lets pidgeonhole the predicted interval into one of graphemes
        predicted_centroid = 0.5 * predicted_s + 0.5 * predicted_e
        # potential_carriers = [i for i in graphalign if (i["time_s"] >= time_s) and (i["time_e"] <= time_e)]
        carriers_centroids = [0.5 * i["time_s"] + 0.5 * i["time_e"] for i in carriers]
        diffs = [(i - predicted_centroid) ** 2 for i in carriers_centroids]
        closest = np.argmin(diffs)
        winner = carriers[closest]
        win_s, win_e = winner["time_s"], winner["time_e"]
        return [[win_s - time_s, win_e - time_s]]
    except Exception:
        return None




preds = preds.with_columns(
    pl.struct(
        """events_pred_pp
              unstress
              stress time_e time_s""".split()
    )
    .map_elements(fix_incorrect)
    .alias("events_pred_pp")
).drop_nulls(subset="events_pred_pp")

preds.write_ndjson(outfile)

2 + 2
