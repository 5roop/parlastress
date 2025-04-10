import polars as pl

PS_Mirna = pl.read_ndjson("../data/input/PS_Mirna/ParlaStress-HR.wo-idx.jsonl")
MP_Mirna = pl.read_ndjson(
    "../data/input/MP_Mirna/MP_Mirna_encoding_stress.wo-idx.jsonl"
)
SLO = pl.read_ndjson("../data/input/SLO/SLO_encoding_stress.wo-idx.jsonl")
RS_Mirna = pl.read_ndjson("../data/input/RS_Mirna/ParlaStress-RS.wo-idx.jsonl")

df = pl.concat([PS_Mirna, MP_Mirna, SLO, RS_Mirna], how="vertical_relaxed")
2 + 2
