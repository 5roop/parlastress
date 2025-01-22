from pathlib import Path
import polars as pl
checkpointdirs = [ i for i in Path(".").glob("model*/checkpoint*") if i.is_dir()]
test_files_PS_HR = pl.read_ndjson("data/input/PS_Mirna/ParlaStress-HR.jsonl").filter(pl.col("split_speaker").eq("test"))["audio_wav"].to_list()
test_files_MP = pl.read_ndjson("data/input/MP/MP_combined_stress.jsonl")["audio_wav"].to_list()
test_files_SLO = pl.read_ndjson("data/input/SLO/SLO_encoding_stress.jsonl")["audio_wav"].to_list()
TG_checkpoints = ["model_primstress_1e-5_20_1/checkpoint-6540"]
TG_labels =  ["event optimal"]




rule gather_textgrids:
    input:
        wavs_SLO=[f"data/SLO_test_textgrids/{Path(i).name}" for i in test_files_SLO],
        tgs_SLO = [f"data/SLO_test_textgrids/{Path(i).with_suffix('.TextGrid').name}" for i in test_files_SLO],
        logs_SLO= [f"data/SLO_test_textgrids/{Path(i).with_suffix('.errors_found').name}" for i in test_files_SLO],
        wavs_MP=[f"data/MP_test_textgrids/{Path(i).name}" for i in test_files_MP],
        tgs_MP = [f"data/MP_test_textgrids/{Path(i).with_suffix('.TextGrid').name}" for i in test_files_MP],
        logs_MP= [f"data/MP_test_textgrids/{Path(i).with_suffix('.errors_found').name}" for i in test_files_MP],
        wavs_PS_HR=[f"data/PS-HR_test_textgrids/{Path(i).name}" for i in test_files_PS_HR],
        tgs_PS_HR = [f"data/PS-HR_test_textgrids/{Path(i).with_suffix('.TextGrid').name}" for i in test_files_PS_HR],
        logs_PS_HR= [f"data/PS-HR_test_textgrids/{Path(i).with_suffix('.errors_found').name}" for i in test_files_PS_HR],
        moot = "model_scores.png",
        words_10_MP = "data/input/MP/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_MP = "data/input/MP/stress_words_wav2vec2_20_epoch.jsonl",
        words_10_PS_HR = "data/input/PS_Mirna/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_PS_HR = "data/input/PS_Mirna/stress_words_wav2vec2_20_epoch.jsonl",
        words_10_SLO = "data/input/SLO/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_SLO = "data/input/SLO/stress_words_wav2vec2_20_epoch.jsonl",
    output:
        log_SLO = "data/SLO_test_textgrids_errors.log",
        log_MP = "data/MP_test_textgrids_errors.log",
        log_PS_HR = "data/PS-HR_test_textgrids_errors.log"
    shell:
        """
        cat {input.logs_SLO} | tee {output.log_SLO}
        cat {input.logs_MP} | tee {output.log_MP}
        cat {input.logs_PS_HR} | tee {output.log_PS_HR}
        rm -rf parlastress
        git clone https://github.com/clarinsi/parlastress.git
        rm -rf parlastress/prim_stress/*/stress_wav2vec*
        mkdir -p parlastress/prim_stress/SLO/stress_wav2vecbert2/
        mkdir -p parlastress/prim_stress/MP/stress_wav2vecbert2/
        mkdir -p parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/

        cp {input.tgs_SLO} parlastress/prim_stress/SLO/stress_wav2vecbert2/
        cp {output.log_SLO} parlastress/prim_stress/SLO/stress_wav2vecbert2.log

        cp {input.tgs_MP} parlastress/prim_stress/MP/stress_wav2vecbert2/
        cp {output.log_MP} parlastress/prim_stress/MP/stress_wav2vecbert2.log

        cp {input.tgs_PS_HR} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/
        cp {output.log_PS_HR} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2.log

        cp {input.words_10_MP} {input.words_20_MP} parlastress/prim_stress/MP
        cp {input.words_10_PS_HR} {input.words_20_PS_HR} parlastress/prim_stress/PS_Mirna
        cp {input.words_10_SLO} {input.words_20_SLO} parlastress/prim_stress/SLO

        cd parlastress
        git add prim_stress/*
        git commit -m "Add word files"
        git push
        cd ..
        rm -rf parlastress
        """
rule generate_jsons_for_nikola_PS:
    conda: "transformers"
    input:
        data = lambda wildcards: {
            "PS_Mirna": ["data/input/PS_Mirna/ParlaStress-HR.jsonl"],
            "MP": ["data/input/MP/MP_encoding_stress.jsonl"],
            "SLO": ["data/input/SLO/SLO_encoding_stress.jsonl"]
        }.get(wildcards.test),
        predictions = lambda wildcards: {
            "10":"model_primstress_1e-5_20_1/checkpoint-3270_postprocessedpredictions.jsonl",
            "20": "model_primstress_1e-5_20_1/checkpoint-6540_postprocessedpredictions.jsonl"
        }.get(wildcards.epoch)
    output: "data/input/{test}/stress_words_wav2vec2_{epoch}_epoch.jsonl"
    script: "scripts/generate_jsons_for_nikola.py"


rule generate_tg_output_PS:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/PS_Mirna/stress/{file}.stress.TextGrid",
        inwav = "data/input/PS_Mirna/wav/{file}.wav",
    output:
        wav = "data/PS-HR_test_textgrids/{file}.wav",
        tg = "data/PS-HR_test_textgrids/{file}.TextGrid",
        error_report = temp("data/PS-HR_test_textgrids/{file}.errors_found")
    params:
        models=TG_checkpoints,
        model_labels=TG_labels
    conda: "praatting"
    script: "scripts/generate_tg_output.py"

rule generate_tg_output_MP:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/MP/split_textgrids/{file}.stress.TextGrid",
        inwav = "data/input/MP/split_wavs/{file}.wav",
    output:
        wav = "data/MP_test_textgrids/{file}.wav",
        tg = "data/MP_test_textgrids/{file}.TextGrid",
        error_report = temp("data/MP_test_textgrids/{file}.errors_found")
    params:
        models=TG_checkpoints,
        model_labels=TG_labels
    conda: "praatting"
    script: "scripts/generate_tg_output.py"
rule generate_tg_output_SLO:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/SLO/stress/{file}.stress.TextGrid",
        inwav = "data/input/SLO/wav/{file}.wav",

    output:
        wav = "data/SLO_test_textgrids/{file}.wav",
        tg = "data/SLO_test_textgrids/{file}.TextGrid",
        error_report = temp("data/SLO_test_textgrids/{file}.errors_found")
    params:
        models=TG_checkpoints,
        model_labels=TG_labels
    conda: "praatting"
    script: "scripts/generate_tg_output.py"
rule gather_stats:
# run with
# export CUDA_VISIBLE_DEVICES=2; snakemake -j 10 --use-conda --conda-frontend mamba  --rerun-incomplete --batch gather_stats=2/10 gather_stats
    input:
        [f"{i}_stats.jsonl" for i in checkpointdirs]
    output: "model_scores.png"
    conda: "transformers"
    script: "scripts/plot_scores.py"

rule calculate_stats:
    input: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_postprocessedpredictions.jsonl"
    output: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_stats.jsonl"
    conda: "transformers"
    params:
        modes = ["raw", "pp"],
        provenances = ["PS-HR", "MP", "SLO"]
    script: "scripts/calculate_stats.py"

rule postprocess_predictions:
    input:
        predictions = "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl",
        datafiles=["data_MP.jsonl", "data_PS-HR.jsonl", "data_SLO.jsonl"]
    output: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_postprocessedpredictions.jsonl"
    conda: "transformers"
    script: "scripts/postprocess_predictions.py"

rule run_inference:
    input:
        checkpoint = "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}/",
        datafiles = ["data_MP.jsonl", "data_PS-HR.jsonl", "data_SLO.jsonl"]
    output:  "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl"
    conda: "transformers"
    script:
        "scripts/run_inference.py"
rule get_data:
    shell:
        """
        bash 0_download_data.sh
        """
