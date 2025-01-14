from pathlib import Path
import polars as pl
checkpointdirs = [ i for i in Path(".").glob("model*/checkpoint*") if i.is_dir()]
test_files = pl.read_ndjson("data/input/PS_Mirna/ParlaStress-HR.jsonl").filter(pl.col("split_speaker").eq("test"))["audio_wav"].to_list()


rule gather_textgrids:
    input:
        wavs=[f"data/PS-HR_test_textgrids/{Path(i).name}" for i in test_files],
        tgs = [f"data/PS-HR_test_textgrids/{Path(i).with_suffix('.TextGrid').name}" for i in test_files],
        logs= [f"data/PS-HR_test_textgrids/{Path(i).with_suffix('.errors_found').name}" for i in test_files],
    output:
        "data/PS-HR_test_textgrids_errors.log"
    shell:
        """cat {input.logs} | tee {output}
        git clone https://github.com/clarinsi/parlastress.git
        rm -rf parlastress/prim_stress/PS_Mirna/stress_wav2vec*
        mkdir -p parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/

        cp {input.tgs} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2/
        cp {output} parlastress/prim_stress/PS_Mirna/stress_wav2vecbert2.log
        cd parlastress
        git add prim_stress/PS_Mirna/stress_wav2vecbert2/ prim_stress/PS_Mirna/stress_wav2vecbert2.log
        git commit
        git push
        cd ..
        rm -rf parlastress
        """

rule generate_tg_output:
    input:
        img = "model_scores.png",
        intextgrid = "data/input/PS_Mirna/stress/{file}.stress.TextGrid",
        inwav = "data/input/PS_Mirna/wav/{file}.wav",
    params:
        models = [
            "model_primstress_8e-6_20_1/checkpoint-654",
            "model_primstress_8e-6_20_1/checkpoint-3924",
            "model_primstress_8e-6_20_1/checkpoint-6540",
            "model_primstress_5e-5_20_4/checkpoint-654"
        ],
        model_labels = [
            "epoch2",
            "epoch12_event_optimal",
            "epoch20",
            "frame_optimal"
        ]
    output:
        wav = "data/PS-HR_test_textgrids/{file}.wav",
        tg = "data/PS-HR_test_textgrids/{file}.TextGrid",
        error_report = temp("data/PS-HR_test_textgrids/{file}.errors_found")
    conda: "praatting"
    script: "scripts/generate_tg_output.py"

rule gather_stats:
# run with
# export CUDA_VISIBLE_DEVICES=5; snakemake -j 1 --use-conda --conda-frontend mamba  --batch gather_stats=3/10
    input:
        [f"{i}_stats.jsonl" for i in checkpointdirs]
    output: "model_scores.png"
    conda: "transformers"
    script: "scripts/plot_scores.py"

rule calculate_stats:
    input: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl"
    output: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_stats.jsonl"
    conda: "transformers"
    script: "scripts/calculate_stats.py"
rule run_inference:
    input: "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}/"
    output:  "model_primstress_{LR}_{epochs}_{gas}/checkpoint-{checkpoint_num}_predictions.jsonl"
    conda: "transformers"
    script:
        "scripts/run_inference.py"
rule get_data:
    shell:
        """
        bash 0_download_data.sh
        """
