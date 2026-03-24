# load  pretrained model from modal
import os
from pdb import run
import torch
import wandb
import pandas as pd

# Plot the data
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from ..model import NeoBERTLMHead, NeoBERTConfig
from ..tokenizer import get_tokenizer
from .analysis import AnalysisTrainedModel
from ..pretraining.trainer import to_target_batch_size
from ..dataset import get_dataset
from ..dataloader import get_dataloader

from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs


import numpy as np

#

# I thus define a wanb workspace
# for every run I analyse, I create a section in the workspace
# in each section I log the  relevant data with a prefix corresponding to the run name
# I have to check how to efficiently log the tables
# and  then I create custom panels in the section to visualise all of these
#


def get_config(base_path):
    cfg_path = os.path.join(base_path, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def extract_wandb_run_id(base_path):
    # Load the wandb logs from the specified base path
    wandb_folder_path = os.path.join(base_path, "wandb")
    for file_name in os.listdir(wandb_folder_path):
        if file_name.startswith("run"):
            run_id = file_name.split("-")[-1]
            print(f"Extracted run ID: {run_id}")
            return run_id
    return None


def load_wandb_history_and_tables_from_file(run_id, project_name="MoP"):
    # Access the run data
    api = wandb.Api()
    id_string = f"{project_name}/{run_id}"
    run = api.run(id_string)

    # Fetch the history data (this returns a pandas DataFrame)
    history = run.history()

    tables_dict = {}

    for artifact in run.logged_artifacts():
        # skip internal stuff like history/events
        if artifact.type not in ["wandb-history", "wandb-events"]:
            for key in artifact.manifest.entries.keys():
                obj = artifact.get(key)
                if isinstance(obj, wandb.data_types.Table):
                    tables_dict[artifact.name] = obj  # store the table itself

    # print the columns of each table for debugging
    for name, table in tables_dict.items():
        print(f"Table '{name}' columns: {table.columns}")

    return history, tables_dict


def plot_wandb_tables(tables_dict, run_name):
    for name, table in tables_dict.items():
        if "max_expert_usage_per_layer" in name:
            log_max_expert_usage_per_layer_plots(table, run_name)
        elif "mean_expert_usage_per_layer" in name:
            log_mean_expert_usage_per_layer_plots(table, run_name)
        elif "expert_usage_proportions" in name:
            log_expert_usage_proportions_plots(table, run_name)


def plot_wandb_history(
    history_df,
    run_name,
    individual_columns_to_log={
        "train/accuracy",
        "train/expert_loss",
        "train/difference_mean_expert_usage_across_layers",
    },
    grouped_columns_to_log={
        "normalised_expert_cost_rel_stds": [
            "train/inter_sequence_normalised_expert_cost_rel_std",
            "train/intra_sequence_normalised_expert_cost_rel_std",
            "train/inter_token_normalised_expert_cost_rel_std",
        ],
        "entropy": [
            "train/entropy",
            "train/entropy_per_sequence",
        ],
    },
):
    # grouped plots
    for group_name, columns in grouped_columns_to_log.items():
        subset_cols = [c for c in columns if c in history_df.columns]
        if subset_cols:
            xs = history_df["_step"].dropna().unique().tolist()
            ys = []
            keys = []
            for col in subset_cols:
                series = history_df[["_step", col]].dropna()
                if not series.empty:
                    ys.append(series[col].tolist())
                    key = col.split("/")[-1]
                    if "std" in key:
                        key = (
                            key.split("_")[0] + "_" + key.split("_")[1]
                        )  # take first part for brevity
                    keys.append(key)
            if ys:
                wandb.log(
                    {
                        f"{run_name}/{group_name}_plot": wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title=group_name,
                            xname="step",
                        )
                    }
                )

    # individual plots
    for column in individual_columns_to_log:
        if column in history_df.columns:
            series = history_df[["_step", column]].dropna()
            if not series.empty:
                xs = history_df["_step"].dropna().unique().tolist()
                ys = [series[column].tolist()]  # wrapped in a list
                keys = [column.split("/")[-1]]
                wandb.log(
                    {
                        f"{column.replace('train', run_name, 1)}_plot": wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title=column.split("/")[-1],
                            xname="step",
                        )
                    }
                )


def log_max_expert_usage_per_layer_plots(max_expert_usage_per_layer_table, run_name):
    """
    Converts a wandb.Table in long format with columns [step, lineVal, lineKey]
    into a wandb line_series plot and logs it.

    Args:
        run: The current wandb run.
        expert_usage_proportions_table (wandb.Table): Table with columns step, lineVal, lineKey
    """
    # Convert wandb.Table to pandas DataFrame
    df = pd.DataFrame(
        max_expert_usage_per_layer_table.data,
        columns=max_expert_usage_per_layer_table.columns,
    )

    # Pivot into wide format: rows=step, columns=lineKey, values=lineVal
    pivot_df = df.pivot(index="step", columns="lineKey", values="lineVal").reset_index()

    # xs = step values
    xs = pivot_df["step"].tolist()

    # ys = list of lists, each corresponding to a LineKey
    ys = [pivot_df[col].tolist() for col in pivot_df.columns if col != "step"]

    # keys = list of expert names
    keys = [col for col in pivot_df.columns if col != "step"]

    # Create wandb line plot
    line_plot = wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="Max Expert Usage Per Layer", xname="Step"
    )

    # Log to wandb
    wandb.log({f"{run_name}/max_expert_usage_per_layer": line_plot})


# def log_max_expert_usage_per_layer_plots(max_expert_usage_per_layer_table, run_name):

#     """
#     Logs a custom Vega line plot to W&B given a wandb.Table in long format
#     with columns [step, lineVal, lineKey].
#     """
#     # Log plot using wandb.plot_table
#     line_plot = wandb.plot_table(
#         vega_spec,
#         max_expert_usage_per_layer_table,
#         fields={"step": "step", "lineVal": "lineVal", "lineKey": "lineKey"},
#         string_fields={"title": "Max Expert Usage Per Layer", "xname": "Step"},
#         split_table=True
#     )

#     wandb.log({f"{run_name}/max_expert_usage_per_layer": line_plot})

# Create a custom chart using a Vega-Lite spec and the data table.


def log_mean_expert_usage_per_layer_plots(mean_expert_usage_per_layer_table, run_name):
    # adapt expert usage proportions code to mean expert usage per layer
    """
    Converts a wandb.Table in long format with columns [step, lineVal, lineKey]
    into a wandb line_series plot and logs it.

    Args:
        run: The current wandb run.
        expert_usage_proportions_table (wandb.Table): Table with columns step, lineVal, lineKey
    """
    # Convert wandb.Table to pandas DataFrame
    df = pd.DataFrame(
        mean_expert_usage_per_layer_table.data,
        columns=mean_expert_usage_per_layer_table.columns,
    )

    # Pivot into wide format: rows=step, columns=lineKey, values=lineVal
    pivot_df = df.pivot(index="step", columns="lineKey", values="lineVal").reset_index()

    # xs = step values
    xs = pivot_df["step"].tolist()

    # ys = list of lists, each corresponding to a LineKey
    ys = [pivot_df[col].tolist() for col in pivot_df.columns if col != "step"]

    # keys = list of expert names
    keys = [col for col in pivot_df.columns if col != "step"]

    # Create wandb line plot
    line_plot = wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="Mean Expert Usage Per Layer", xname="Step"
    )
    # Log to wandb
    wandb.log({"mean_expert_usage_per_layer": line_plot})


# def log_mean_expert_usage_per_layer_plots(mean_expert_usage_per_layer_table, run_name,vega_spec):

#     """
#     Logs a custom Vega line plot to W&B given a wandb.Table in long format
#     with columns [step, lineVal, lineKey].
#     """
#     # Log plot using wandb.plot_table
#     line_plot = wandb.plot_table(
#         vega_spec,
#         mean_expert_usage_per_layer_table,
#         fields={"step": "step", "lineVal": "lineVal", "lineKey": "lineKey"},
#         string_fields={"title": "Mean Expert Usage Per Layer", "xname": "Step"},
#         split_table=False
#     )

#     wandb.log({"mean_expert_usage_per_layer": line_plot})


def log_expert_usage_proportions_plots(expert_usage_proportions_table, run_name):
    """
    Converts a wandb.Table in long format with columns [step, lineVal, lineKey]
    into a wandb line_series plot and logs it.

    Args:
        run: The current wandb run.
        expert_usage_proportions_table (wandb.Table): Table with columns step, lineVal, lineKey
    """
    # Convert wandb.Table to pandas DataFrame
    df = pd.DataFrame(
        expert_usage_proportions_table.data,
        columns=expert_usage_proportions_table.columns,
    )

    # Pivot into wide format: rows=step, columns=lineKey, values=lineVal
    pivot_df = df.pivot(index="step", columns="lineKey", values="lineVal").reset_index()

    # xs = step values
    xs = pivot_df["step"].tolist()

    # ys = list of lists, each corresponding to a LineKey
    ys = [pivot_df[col].tolist() for col in pivot_df.columns if col != "step"]

    # keys = list of expert names
    keys = [col for col in pivot_df.columns if col != "step"]

    # Create wandb line plot
    line_plot = wandb.plot.line_series(
        xs=xs, ys=ys, keys=keys, title="Expert Usage Proportions", xname="Step"
    )

    # Log to wandb
    wandb.log({"expert_usage_proportions": line_plot})


# def log_expert_usage_proportions_plots(expert_usage_proportions_table, run_name, vega_spec):
#     """
#     Logs a custom Vega line plot to W&B given a wandb.Table in long format
#     with columns [step, lineVal, lineKey].
#     """
#     # Log plot using wandb.plot_table
#     line_plot = wandb.plot_table(
#         vega_spec,
#         expert_usage_proportions_table,
#         fields={"step": "step", "lineVal": "lineVal", "lineKey": "lineKey"},
#         string_fields={"title": "Expert Usage Proportions", "xname": "Step"},
#         split_table=False
#     )
#     wandb.log({"expert_usage_proportions": line_plot})


def load_pretrained_models_modal(base_path, checkpoint):
    cfg = get_config(base_path)
    tokenizer = get_tokenizer(**cfg.tokenizer)
    neobert_model = NeoBERTLMHead(
        NeoBERTConfig(**cfg.model, **cfg.tokenizer, pad_token_id=tokenizer.pad_token_id)
    )
    state_dict_path = os.path.join(
        base_path, "model_checkpoints", checkpoint, "state_dict.pt"
    )
    neobert_state_dict = torch.load(state_dict_path, map_location="cpu")

    # Fix keys: strip "_orig_mod." if present
    new_state_dict = {}
    for k, v in neobert_state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v

    neobert_model.load_state_dict(new_state_dict, strict=True)
    return neobert_model


def set_up_acceleration(cfg_analysis):
    # set up the accelerator to run the analysis
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,  # enable manual control of the scheduler
        mixed_precision=cfg_analysis.trainer.mixed_precision,
        gradient_accumulation_steps=cfg_analysis.trainer.gradient_accumulation_steps,
        log_with="wandb",
        # project_config=project_config,
        kwargs_handlers=[kwargs],
    )
    # Initialise the wandb run and pass wandb parameters
    os.makedirs(cfg_analysis.wandb.dir, exist_ok=True)
    accelerator.init_trackers(
        project_name=cfg_analysis.wandb.project,
        init_kwargs={
            "wandb": {
                "name": "test_wandb",  # cfg_analysis.wandb.name
                "entity": cfg_analysis.wandb.entity,
                "config": OmegaConf.to_container(cfg_analysis)
                | {"distributed_type": accelerator.distributed_type},
                "tags": cfg_analysis.wandb.tags,
                "dir": cfg_analysis.wandb.dir,
                "mode": cfg_analysis.wandb.mode,
                "resume": cfg_analysis.wandb.resume,
            }
        },
    )

    set_seed(cfg_analysis.seed)

    return accelerator


def analyse_pretrained_model(base_path, checkpoint, cfg_analysis, max_steps=10):  # 1000

    # set up the accelerator to run the analysis
    accelerator = set_up_acceleration(cfg_analysis)

    # Enable TF32 on matmul and on cuDNN
    torch.backends.cuda.matmul.allow_tf32 = cfg_analysis.trainer.tf32
    torch.backends.cudnn.allow_tf32 = cfg_analysis.trainer.tf32

    # Get the dtype for the pad_mask
    dtype_pad_mask = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype_pad_mask = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype_pad_mask = torch.bfloat16

    cfg = get_config(base_path)  # the pretrained model config
    model_run_name = f"b_strt: {cfg.model.loss.cost_based_loss_alpha_start:.1e}_a_end: {cfg.model.loss.cost_based_loss_alpha_end:.1e}_scaling: {cfg.model.loss.alpha_scaling}_cst_exp:{cfg.model.expert_cost_exponent}"
    wandb_run_id = extract_wandb_run_id(base_path)
    wandb_history_df, wandb_tables_dict = load_wandb_history_and_tables_from_file(
        wandb_run_id
    )

    plot_wandb_history(wandb_history_df, run_name=model_run_name)
    plot_wandb_tables(wandb_tables_dict, run_name=model_run_name)

    model = load_pretrained_models_modal(base_path, checkpoint)
    tokenizer = get_tokenizer(**cfg.tokenizer)

    test_dataset = get_dataset(cfg=cfg, hf_path="JeanKaddour/minipile", split="test")

    dataloader = get_dataloader(
        test_dataset,
        tokenizer,
        dtype=dtype_pad_mask,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        batch_size=cfg.dataloader.train.batch_size,
        **cfg.datacollator,
    )  # ultimately should be moved to cfg.dataloader.test

    # Prepare everything with the accelerator: place the model and dataloader on the right device
    # and cast to the right dtype
    (
        dataloader,
        model,
    ) = accelerator.prepare(
        dataloader,
        model,
    )
    model.eval()
    analysetrained = AnalysisTrainedModel(cfg, accelerator, max_steps)
    step = 0
    for batch in dataloader:
        step += 1
        if step >= max_steps:
            break
        model_output = model(
            batch["input_ids"],
            batch.get("attention_mask", None),
            output_expert_usage_loss=True,
            output_router_logits=True,
        )
        analysetrained(batch, model_output, step)


def analyse_list_of_pretrained_models(
    list_of_base_paths=["/runs/logs/checkpoints/mop_2025-09-29_15-28-17"],
):  # "/runs/logs/checkpoints/mop_2025-09-29_15-27-29"

    for base_path in list_of_base_paths:
        print(f"Analysing model at {base_path}")
        cfg = get_config(base_path)
        analyse_pretrained_model(base_path, checkpoint="100", cfg_analysis=cfg)


# créer cfg
