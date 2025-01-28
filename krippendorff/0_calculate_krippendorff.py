from sklearn.metrics import accuracy_score
import polars as pl
from pathlib import Path
import krippendorff

first_file = "data/input/MP_Mirna/MP_Mirna_encoding_stress.wo-idx.jsonl"
second_file = "data/input/MP/MP_encoding_stress.wo-idx.jsonl"

assert Path(first_file).exists()
assert Path(second_file).exists()
pl.Config.set_tbl_cols(n=-1)
pl.Config.set_fmt_str_lengths(10)
pl.Config.set_tbl_width_chars(800)
pl.Config.set_tbl_rows(None)


def read_file(file):
    df = (
        pl.read_ndjson(file)
        .with_columns(
            pl.lit("test").alias("split_speaker"),
        )
        .explode(["multisyllabic_words"])
        .unnest("multisyllabic_words")
        .with_columns(
            pl.col("stress").list.len().alias("stress_list_length"),
        )
        .explode("stress")
    )

    if not df.filter(pl.col("stress_list_length") > 1).shape[0] == 0:
        print("There are words with more than one stress, dropping!")
        df = df.filter(pl.col("stress_list_length") <= 1)
    with pl.Config(fmt_str_lengths=15, tbl_cols=25):
        print("Will drop these instances:")
        print(df.filter(pl.any_horizontal(pl.all().is_null())))
    df = df.drop_nulls()

    df = df.with_columns(
        pl.col("audio_wav").alias("audio"),
    ).with_columns(
        pl.lit("MP").alias("provenance"),
    )

    def get_syllable_idx(row):
        candidates = [row["stress"], *row["unstress"]]
        candidates = sorted(candidates, key=lambda d: float(d["time_s"]))
        for i, c in enumerate(candidates):
            if c == row["stress"]:
                return i

    df = df.with_columns(
        pl.struct(["stress", "unstress"])
        .map_elements(get_syllable_idx, return_dtype=pl.Int8)
        .alias("stressed_syllable_idx")
    )
    # return df.select(
    #     "id chapter audio_start audio_end text words word time_s time_e stress".split()
    # )
    return df


df1 = read_file(first_file)
df2 = read_file(second_file)

combo = df1.join(
    df2,
    on="id chapter audio_start audio_end text words word time_s time_e".split(),
    how="inner",
)


# def check_for_overlap(row):
#     this, other = row["stressed_syllable_idx"], row["stressed_syllable_idx_right"]
#     ts, te = this["time_s"], this["time_e"]
#     os, oe = other["time_s"], other["time_e"]
#     return (ts < oe) and (te > os)


# combo = combo.with_columns(
#     pl.struct(["stress", "stress_right"])
#     .map_elements(check_for_overlap, return_dtype=pl.Boolean)
#     .alias("matches")
# )

y1 = combo["stressed_syllable_idx"].to_list()
y2 = combo["stressed_syllable_idx_right"].to_list()

print(krippendorff.alpha(reliability_data=[y1, y2], level_of_measurement="nominal"))
print(accuracy_score(y1, y2), accuracy_score(y2, y1))
2 + 2
