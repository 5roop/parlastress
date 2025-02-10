import polars as pl

filedict = {
    "ParlaSpeech-HR": ["data/input/PS_Mirna/ParlaStress-HR.jsonl"],
    "ParlaSPeech-SR": ["data/input/RS_Mirna/ParlaStress-RS.jsonl"],
    "MićiPrinc-CKM": ["data/input/MP/MP_encoding_stress.jsonl"],
    "Artur-SL": ["data/input/SLO/SLO_encoding_stress.jsonl"],
}


def read_file(name, file):
    df = (
        pl.read_ndjson(file)
        .explode(["multisyllabic_words"])
        .unnest("multisyllabic_words")
        .with_columns(
            pl.col("stress").list.len().alias("stress_list_length"),
        )
        .explode("stress")
    )
    if not "split_speaker" in df.columns:
        df = df.with_columns(pl.lit("test").alias("split_speaker"))

    if not df.filter(pl.col("stress_list_length") > 1).shape[0] == 0:
        # print("There are words with more than one stress, dropping!")
        df = df.filter(pl.col("stress_list_length") <= 1)
    with pl.Config(fmt_str_lengths=15, tbl_cols=25):
        # print("Will drop these instances:")
        # print(df.filter(pl.any_horizontal(pl.all().is_null())))
        2 + 2
    df = df.drop_nulls()

    df = df.with_columns(
        pl.col("audio_wav").alias("audio"),
    ).with_columns(
        pl.lit(name).alias("provenance"),
    )

    def get_syllable_idx(row):
        candidates = [row["stress"], *row["unstress"]]
        candidates = sorted(candidates, key=lambda d: float(d["time_s"]))
        for i, c in enumerate(candidates):
            if c == row["stress"]:
                return i

    def get_syllable_count(row):
        candidates = [row["stress"], *row["unstress"]]
        return len(candidates)

    df = df.with_columns(
        pl.struct(["stress", "unstress"])
        .map_elements(get_syllable_idx, return_dtype=pl.Int8)
        .alias("stressed_syllable_idx"),
        pl.struct(["stress", "unstress"])
        .map_elements(get_syllable_count, return_dtype=pl.Int32)
        .alias("syllable_count"),
    )
    if not "speaker_id" in df.columns:
        df = df.with_columns(pl.col("audio_wav").alias("speaker_id"))
    df = df.with_columns((pl.col("time_e") - pl.col("time_s")).alias("word_duration"))
    return df.select(
        ["split_speaker", "syllable_count", "speaker_id", "provenance", "word_duration"]
    )


dfs = []
for name, file in filedict.items():
    print(name)
    dfs.append(read_file(name, file))
    df = pl.concat(dfs, how="vertical_relaxed")

print(
    df.group_by("split_speaker provenance".split())
    .agg(
        pl.col("syllable_count").sum().alias("Syllables"),
        pl.col("syllable_count").count().alias("Words"),
        pl.col("speaker_id").n_unique().alias("Speakers"),
        pl.col("word_duration").sum().alias("Duration_s"),
    )
    .sort(["split_speaker", "provenance"], descending=True)
)

2 + 2
