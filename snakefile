from pathlib import Path
import polars as pl
checkpointdirs = [ i for i in Path(".").glob("model*/checkpoint*") if i.is_dir()]
test_files_PS_HR = pl.read_ndjson("data/input/PS_Mirna/ParlaStress-HR.jsonl").filter(pl.col("split_speaker").eq("test"))["audio_wav"].to_list()
test_files_PS_RS = pl.read_ndjson("data/input/RS_Mirna/ParlaStress-RS.jsonl")["audio_wav"].to_list()
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
        wavs_PS_RS=[f"data/PS-RS_test_textgrids/{Path(i).name}" for i in test_files_PS_RS],
        tgs_PS_RS = [f"data/PS-RS_test_textgrids/{Path(i).with_suffix('.TextGrid').name}" for i in test_files_PS_RS],
        logs_PS_RS= [f"data/PS-RS_test_textgrids/{Path(i).with_suffix('.errors_found').name}" for i in test_files_PS_RS],
        moot = "model_scores.png",
        words_10_MP = "data/input/MP/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_MP = "data/input/MP/stress_words_wav2vec2_20_epoch.jsonl",
        words_10_PS_HR = "data/input/PS_Mirna/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_PS_HR = "data/input/PS_Mirna/stress_words_wav2vec2_20_epoch.jsonl",
        words_10_SLO = "data/input/SLO/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_SLO = "data/input/SLO/stress_words_wav2vec2_20_epoch.jsonl",
        words_10_RS = "data/input/RS_Mirna/stress_words_wav2vec2_10_epoch.jsonl",
        words_20_RS = "data/input/RS_Mirna/stress_words_wav2vec2_20_epoch.jsonl",
        pdf_10 = "paper_images/CM_10_epoch.pdf",
        pdf_20 = "paper_images/CM_20_epoch.pdf",
        pdf_average = "paper_images/CM_1200X1000.pdf",
        pdf_learning_curve = "paper_images/learning_curves.pdf"

    output:
        log_SLO = "data/SLO_test_textgrids_errors.log",
        log_MP = "data/MP_test_textgrids_errors.log",
        log_PS_HR = "data/PS-HR_test_textgrids_errors.log",
        log_PS_RS = "data/PS-RS_test_textgrids_errors.log"
    shell:
        """
        cat {input.logs_SLO} > {output.log_SLO}
        cat {input.logs_MP} > {output.log_MP}
        cat {input.logs_PS_HR} > {output.log_PS_HR}
        cat {input.logs_PS_RS} > {output.log_PS_RS}
        rm -rf parlastress
        git clone https://github.com/clarinsi/parlastress.git
        rm -rf parlastress/prim_stress/*/stress_wav2vec*
        mkdir -p parlastress/prim_stress/SLO/stress_wav2vecbert2/
        mkdir -p parlastress/prim_stress/MP/stress_wav2vecbert2/
        mkdir -p parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/
        mkdir -p parlastress/prim_stress/RS_Mirna/stress_wav2vecbert2/

        cp {input.tgs_SLO} parlastress/prim_stress/SLO/stress_wav2vecbert2/
        cp {output.log_SLO} parlastress/prim_stress/SLO/stress_wav2vecbert2.log

        cp {input.tgs_MP} parlastress/prim_stress/MP/stress_wav2vecbert2/
        cp {output.log_MP} parlastress/prim_stress/MP/stress_wav2vecbert2.log

        cp {input.tgs_PS_HR} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/
        cp {output.log_PS_HR} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2.log

        cp {input.tgs_PS_RS} parlastress/prim_stress/RS_Mirna/stress_wav2vecbert2/
        cp {output.log_PS_RS} parlastress/prim_stress/RS_Mirna/stress_wav2vecbert2.log

        cp {input.words_10_RS} {input.words_20_RS} parlastress/prim_stress/RS_Mirna
        cp {input.words_10_PS_HR} {input.words_20_PS_HR} parlastress/prim_stress/PS_Mirna
        cp {input.words_10_SLO} {input.words_20_SLO} parlastress/prim_stress/SLO
        cp {input.words_10_MP} {input.words_20_MP} parlastress/prim_stress/MP
        cp {input.pdf_10} {input.pdf_20} {input.pdf_average} {input.pdf_learning_curve} parlastress/prim_stress/
        cd parlastress
        git add prim_stress/*
        git commit -m "Minor edits to learning_curves.pdf and confusion matrices"
        git push
        cd ..
        rm -rf parlastress
        """
rule get_learning_curve_plot:
    output: "paper_images/learning_curves.pdf"
    shell:
        """cd learning_curves
        echo "*** Running snakemake in learning_curves/ ! ***"
        snakemake -j 1 --use-conda
        """

rule generate_jsons_for_nikola_PS_for_CM:
    conda: "transformers"
    input:
        data =  ["data/input/RS_Mirna/ParlaStress-RS.jsonl",
            "data/input/PS_Mirna/ParlaStress-HR.jsonl",
            "data/input/MP/MP_encoding_stress.jsonl",
            "data/input/SLO/SLO_encoding_stress.jsonl"],
        predictions = lambda wildcards: {
            "10_epoch":"model_primstress_1e-5_20_1/checkpoint-3270_postprocessedpredictions.jsonl",
            "20_epoch": "model_primstress_1e-5_20_1/checkpoint-6540_postprocessedpredictions.jsonl",
            "1200X1000": "learning_curves/models/model_1000_1200_7/checkpoint-1200_postprocessedpredictions.jsonl",
        }.get(wildcards.epoch)
    output: "data/input/indices_{epoch}.jsonl"
    script: "scripts/generate_jsons_for_nikola.py"


rule generate_jsons_for_nikola:
    conda: "transformers"
    input:
        data = lambda wildcards: {
            "PS_Mirna":["data/input/PS_Mirna/ParlaStress-HR.jsonl"],
            "RS_Mirna":["data/input/RS_Mirna/ParlaStress-RS.jsonl"],
            "MP": ["data/input/MP/MP_encoding_stress.jsonl"],
            "SLO": ["data/SLO/SLO_encoding_stress.jsonl"],

        }.get(wildcards.dataset),

        predictions = lambda wildcards: {
            "10_epoch":"model_primstress_1e-5_20_1/checkpoint-3270_postprocessedpredictions.jsonl",
            "20_epoch": "model_primstress_1e-5_20_1/checkpoint-6540_postprocessedpredictions.jsonl",
            "1200X1000": "learning_curves/models/model_1000_1200_7/checkpoint-1200_postprocessedpredictions.jsonl",
        }.get(wildcards.epoch)
    output: "data/input/{dataset}/stress_words_wav2vec2_{epoch}.jsonl"
    script: "scripts/generate_jsons_for_nikola.py"

rule plot_CM:
    conda: "transformers"
    input: ["data/input/indices_{epoch}.jsonl"]
    output: "paper_images/CM_{epoch}.pdf"
    script: "scripts/confusion_matrices.py"

rule generate_tg_output_PS_HR:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/PS_Mirna/stress/{file}.stress.TextGrid",
        inwav = "data/input/PS_Mirna/wav/{file}.wav",
    output:
        wav = "data/PS-HR_test_textgrids/{file}.wav",
        tg = "data/PS-HR_test_textgrids/{file}.TextGrid",
        error_report = "data/PS-HR_test_textgrids/{file}.errors_found",
    params:
        models=TG_checkpoints,
        model_labels=TG_labels
    conda: "praatting"
    script: "scripts/generate_tg_output.py"
rule generate_tg_output_PS_RS:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/RS_Mirna/stress/{file}.stress.TextGrid",
        inwav = "data/input/RS_Mirna/wav/{file}.wav",
    output:
        wav = "data/PS-RS_test_textgrids/{file}.wav",
        tg = "data/PS-RS_test_textgrids/{file}.TextGrid",
        error_report = "data/PS-RS_test_textgrids/{file}.errors_found",
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
        error_report = "data/MP_test_textgrids/{file}.errors_found"
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
        error_report = "data/SLO_test_textgrids/{file}.errors_found"
    params:
        models=TG_checkpoints,
        model_labels=TG_labels
    conda: "praatting"
    script: "scripts/generate_tg_output.py"
rule gather_stats:
# run with
# export CUDA_VISIBLE_DEVICES=1; snakemake -j 10 --use-conda --conda-frontend mamba  --rerun-incomplete --batch gather_stats=1/3 gather_stats -k
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
        provenances = ["PS-HR", "MP", "SLO", "PS-RS"]
    script: "scripts/calculate_stats.py"

rule postprocess_predictions:
    input:
        predictions = "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl",
        datafiles=["data_MP.jsonl", "data_PS-HR.jsonl", "data_SLO.jsonl", "data_PS-RS.jsonl"]
    output: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_postprocessedpredictions.jsonl"
    conda: "transformers"
    script: "scripts/postprocess_predictions.py"

rule run_inference:
    input:
        checkpoint = "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}/",
        datafiles = ["data_MP.jsonl", "data_PS-HR.jsonl", "data_SLO.jsonl", "data_PS-RS.jsonl"]
    output:  "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl"
    conda: "transformers"
    script:
        "scripts/run_inference.py"
rule get_data:
    shell:
        """
        bash 0_download_data.sh
        """
