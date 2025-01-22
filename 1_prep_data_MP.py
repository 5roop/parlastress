import polars as pl
from pathlib import Path
import os
from ripostiglio import SAMPLE_RATE, events_to_frames, FRAME_RATE
from datasets import Audio, Dataset

from pydub import AudioSegment

pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(None)
pl.Config.set_tbl_width_chars(200)
segment_path = Path("data/segments")
segment_path.mkdir(exist_ok=True)
# Reading PS_Mirna:
wavbasepath = "data/input/MP/split_wavs/"
df = (
    pl.read_ndjson("data/input/MP/MP_combined_stress.jsonl")
    .with_columns(
        (
            pl.lit(wavbasepath)
            + pl.col("audio_wav").map_elements(lambda s: Path(s).name)
        ).alias("audio_wav"),
        pl.lit("test").alias("split_speaker"),
    )
    .explode(["multisyllabic_words"])
    .unnest("multisyllabic_words")
    .with_columns(
        pl.col("stress").list.len().alias("stress_list_length"),
    )
    .explode("stress")
    # .select("audio_wav time_s time_e stress split_speaker".split())
)

if not df.filter(pl.col("stress_list_length") > 1).shape[0] == 0:
    print("There are words with more than one stress, dropping!")
    df = df.filter(pl.col("stress_list_length") <= 1)
with pl.Config(fmt_str_lengths=100, tbl_cols=10):
    print("Will drop these instances:")
    print(df.filter(pl.any_horizontal(pl.all().is_null())))
df = df.drop_nulls()


def f(row):
    frames = events_to_frames(
        segment_start=row["time_s"],
        segment_end=row["time_e"],
        stress_start=row["stress"]["time_s"],
        stress_end=row["stress"]["time_e"],
    )
    return frames.tolist()


df = (
    df.with_columns(
        pl.struct(["time_s", "time_e", "stress"]).map_elements(f).alias("label"),
        pl.col("audio_wav").alias("audio"),
    )
    # .select(["audio", "label", "time_s", "time_e", "split_speaker", "stress"])
    .with_columns(
        pl.struct(["audio", "time_s", "time_e"])
        .map_elements(
            lambda row: f"{segment_path / str(Path(row['audio']).with_suffix('').name)}_{row['time_s']:0.2f}_{row['time_e']:0.2f}.wav"
        )
        .alias("segment_name"),
        pl.lit("MP").alias("provenance"),
    )
)
from tqdm import tqdm

df.write_ndjson("data_MP.jsonl")
for recording in tqdm(df["audio"].unique().to_list()):
    subset = df.filter(pl.col("audio").eq(recording))
    audio = AudioSegment.from_wav(recording)
    for row in subset.iter_rows(named=True):
        savepath = row["segment_name"]
        s = int(1000 * row["time_s"])
        e = int(1000 * row["time_e"])
        audio[s:e].export(savepath)

2 + 2
