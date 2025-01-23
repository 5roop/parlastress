try:
    data = snakemake.input.data
    predictions = snakemake.input.predictions
    output = snakemake.output[0]
except NameError:
    data = [
        # "data/input/PS_Mirna/ParlaStress-HR.jsonl",
        # "data/input/MP/MP_encoding_stress.jsonl",
        "data/input/SLO/SLO_encoding_stress.jsonl",
    ]
    predictions = (
        "model_primstress_1e-5_20_1/checkpoint-6540_postprocessedpredictions.jsonl"
    )
    output = "brisi.jsonl"

import polars as pl
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


def overlaps(this, other):
    return (this[0] < other[1]) and (this[1] > other[0])


jdata = [json.loads(line) for line in Path(data[0]).read_text().splitlines()]
for i, item in enumerate(jdata):
    jdata[i]["keyer"] = Path(item["audio_wav"]).with_suffix("").name
preds = pl.read_ndjson(predictions).with_columns(
    pl.col("audio").map_elements(lambda s: Path(s).with_suffix("").name).alias("keyer")
)
keys_in_data = list(set([i["keyer"] for i in jdata]))

# preds = preds.filter(pl.col("provenance").eq("PS-HR"))
print(f"{preds.shape=}")

results = []
for row in tqdm(preds.iter_rows(named=True), total=preds.shape[0]):
    time_s, time_e = row["time_s"], row["time_e"]
    true = [round(i + time_s, 2) for i in row["events_true"][0]]
    jdatasubset = [i for i in jdata if row["keyer"].startswith(i["keyer"])]
    if jdatasubset == []:
        continue
    multisyllabic_words = jdatasubset[0]["multisyllabic_words"]
    word = [
        i for i in multisyllabic_words if overlaps(true, [i["time_s"], i["time_e"]])
    ][0]
    pred_char_idx, true_char_idx = None, None
    candidates = word["stress"] + word["unstress"]
    candidates = sorted(candidates, key=lambda i: float(i["time_s"]))
    for i, candidate in enumerate(candidates):
        if overlaps(true, [candidate["time_s"], candidate["time_e"]]):
            true_char_idx = candidate["char_idx"]
    assert true_char_idx is not None, "Could not find true index!"
    try:
        pred = [round(i + time_s, 2) for i in row["events_pred_pp"][0]]
        for i, candidate in enumerate(candidates):
            if overlaps(pred, [candidate["time_s"], candidate["time_e"]]):
                pred_char_idx = candidate["char_idx"]
        assert pred_char_idx is not None, "Could not find predicted index!"
    except IndexError:
        pred_char_idx = None
    results.append(
        {
            "id": jdatasubset[0]["id"],
            "file_name": jdatasubset[0]["audio_wav"],
            "word": word["word"],
            "true_char_idx": true_char_idx,
            "pred_char_idx": pred_char_idx,
            "nuclei": [i["char_idx"] for i in candidates],
        }
    )

pl.DataFrame(results).write_ndjson(output)

print(pl.DataFrame(results).shape)
