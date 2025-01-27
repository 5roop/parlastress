try:
    snakemake.input
except NameError:
    svm_file = "data/input/MP/feat4_MP_svm_results.jsonl"
    transformer_file = "data/input/MP/stress_words_wav2vec2_10_epoch.jsonl"
    master_file = "data/input/MP/MP_encoding_stress.jsonl"

import polars as pl

pl.Config.set_tbl_cols(n=-1)
pl.Config.set_fmt_str_lengths(10)
pl.Config.set_tbl_width_chars(800)
pl.Config.set_tbl_rows(None)


master = pl.read_ndjson(master_file)
if "split_speaker" in master.columns:
    master = master.filter(pl.col("split_speaker").eq("test"))
ids = master.select("id").unique(maintain_order=True)["id"].to_list()

svm = pl.read_ndjson(svm_file)
tr = pl.read_ndjson(transformer_file)
assert svm.shape[0] == tr.shape[0], (
    "Row mismatch between SVM and transformer predictions"
)


def reorder_df(df):
    df = pl.concat([df.filter(pl.col("id").eq(id)) for id in ids])
    return df


tr = reorder_df(tr)
svm = reorder_df(svm)
assert svm["id"].to_list() == tr["id"].to_list()
assert svm["audio"].to_list() == tr["audio"].to_list()
assert svm["word"].to_list() == tr["word"].to_list()
assert svm["true_char_idx"].to_list() == tr["true_char_idx"].to_list()
svm = svm.unnest("pred_char_idx")

df = pl.concat([tr, svm.select(pl.selectors.starts_with("model"))], how="horizontal")

wrongs = df.filter(pl.col("true_char_idx") != pl.col("pred_char_idx"))
print(f"Stats: #words: {wrongs.shape[0]}, #nuclei: {wrongs['nuclei'].list.len().sum()}")
for col in wrongs.select(pl.selectors.starts_with("model_")).columns:
    TPS = (wrongs["true_char_idx"] == wrongs[col]).sum()
    acc = TPS / wrongs.shape[0]

    print(
        f"Accuracy on wrongs for {col=}: {acc:0.3f} . Better than transformers on {TPS} words"
    )

2 + 2
