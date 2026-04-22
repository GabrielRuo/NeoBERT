# import subprocess
import sys
from pathlib import Path

import modal
import wandb
import yaml
from omegaconf import OmegaConf
import shutil
from hydra import initialize, compose

import os

TIMEOUT = 24 * 60 * 60  # one day

# Keep Modal image compatible with segmented optional dependencies in pyproject.toml.
MODAL_OPTIONAL_DEPS = ["modal", "train"]

app = modal.App("neobert")

# use os.getenv to get the secrets from the environment variables that were loaded from the .env file.
hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY")

# Pass it to Modal
huggingface_secret = modal.Secret.from_dict({"HF_TOKEN": hf_token})
wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": wandb_api_key})


# experiment_image = (
#     modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04")
#     .apt_install("python3", "python3-pip", "python3-dev", "build-essential", "git", "libaio-dev")
#     .run_commands("update-alternatives --install /usr/bin/python python /usr/bin/python3 1")
#     .pip_install_from_pyproject(Path(__file__).resolve().parents[2] / "pyproject.toml")
#     .add_local_dir("../../conf", "/conf")  # Add the configuration directory

# )

# experiment_image = (
#     modal.Image.debian_slim()
#     .apt_install("git")
#     .run_commands("pip install deepspeed==0.15.4 --global-option=--no_cuda")
#     .pip_install_from_pyproject(Path(__file__).resolve().parents[2] / "pyproject.toml")
#     .env({"DS_BUILD_OPS": "0"})
#     .add_local_dir("../../conf", "/conf")
# )

experiment_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_pyproject(
        Path(__file__).resolve().parents[2] / "pyproject.toml",
        optional_dependencies=MODAL_OPTIONAL_DEPS,
    )
    .add_local_dir(str(Path(__file__).resolve().parents[2] / "conf"), "/root/conf")
)


runs_volume = modal.Volume.from_name("neobert-runs", create_if_missing=True)
training_data_volume = modal.Volume.from_name(
    "neobert-training-data", create_if_missing=True
)
# config_volume = modal.Volume.from_name("neobert-config", create_if_missing=True)

# config_volume = modal.Mount.from_local_dir(
#     "../../conf",  # Local path on your machine
#     remote_path="/workspace/conf",  # Path inside the container
# )


# with experiment_image.imports():
#     from neobert.pretraining import trainer
#     from hydra import initialize, compose


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",  # SWITCH TO T4 if want cheaper
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
# def run_pretrain(args: list[str]) -> None:
#     """Launch pretraining using the provided command arguments."""
#     subprocess.run(args, check=True)
def run_pretrain(overrides: list[str], model_type: str) -> None:
    """Launch pretraining using the provided command arguments."""
    from neobert.pretraining import trainer

    with initialize(config_path="../conf", version_base=None):
        config_name = "pretraining_" + model_type
        cfg = compose(config_name=config_name, overrides=overrides)
        trainer(cfg)


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_predictor(overrides: list[str]) -> None:
    """Launch predict routing using the provided command arguments."""
    from neobert.predictor import predictor

    with initialize(config_path="../conf", version_base=None):
        config_name = "predictor"
        cfg_predictor = compose(config_name=config_name, overrides=overrides)
        predictor(cfg_predictor)


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_pathways_analysis(overrides: list[str]) -> None:
    """Launch pathways analysis using the provided command arguments."""
    from neobert.analysis.pathways import pathways_analysis

    with initialize(config_path="../conf", version_base=None):
        config_name = "predictor"
        cfg_predictor = compose(config_name=config_name, overrides=overrides)
        pathways_analysis(cfg_predictor)


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_difficulty(overrides: list[str]) -> None:
    """Launch difficulty measurement using the provided command arguments."""
    from neobert.difficulty import measure_difficulty

    with initialize(config_path="../conf", version_base=None):
        config_name = "difficulty"
        cfg_dif = compose(config_name=config_name, overrides=overrides)
        measure_difficulty(cfg_dif)

@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_analyses_pretrained() -> None:
    from neobert.analysis.analyse_pretrained import analyse_list_of_pretrained_models

    analyse_list_of_pretrained_models()


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_pretrain_sweep(overrides: list[str], model_type: str) -> None:
    """Launch pretraining using the provided command arguments."""
    from neobert.pretraining import train_and_eval_sweep

    with initialize(config_path="../conf", version_base=None):
        config_name = "pretraining_" + model_type
        cfg = compose(config_name=config_name, overrides=overrides)
        with open("conf/sweeps/sweep.yaml") as f:
            sweep_cfg = yaml.safe_load(f)

        def sweep_main():
            train_and_eval_sweep(cfg)

        sweep_id = wandb.sweep(sweep=sweep_cfg, project="MoP")
        wandb.agent(sweep_id=sweep_id, function=sweep_main, count=3)


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_glue(overrides: list[str]) -> None:
    """Launch finetuning using the provided command arguments."""
    from neobert.glue import trainer

    with initialize(config_path="../conf", version_base=None):
        config_name = "glue_mop"
        cfg_glue = compose(config_name=config_name, overrides=overrides)
        trainer(cfg_glue)


@app.function(image=experiment_image, volumes={"/runs": runs_volume})
def delete_in_volume(remote_dir: str):
    path = Path("/runs") / remote_dir
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted {path}")
    else:
        print(f"Path not found: {path}")


@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_pretrained_model_tester(overrides: list[str]) -> None:
    """Launch testing of pretrained models using the provided command arguments."""
    from neobert.analysis import pretrained_model_tester

    with initialize(config_path="../conf", version_base=None):
        config_name = "tester"
        cfg_tester = compose(config_name=config_name, overrides=overrides)
        pretrained_model_tester(cfg_tester)

@app.function(
    image=experiment_image,
    secrets=[wandb_secret, huggingface_secret],
    gpu="T4",
    volumes={
        "/data": training_data_volume,
        "/runs": runs_volume,
        # "/conf": config_volume,
    },
    timeout=TIMEOUT,
)
def run_visualise_pathways(overrides: list[str]) -> None:
    """Launch pathways analysis using the provided command arguments."""
    
    from neobert.analysis.pathways_visualiser import visualise_pathways

    with initialize(config_path="../conf", version_base=None):
        config_name = "pathways_visualiser"
        cfg_pathways_visualiser = compose(config_name=config_name, overrides=overrides)
        visualise_pathways(cfg_pathways_visualiser)