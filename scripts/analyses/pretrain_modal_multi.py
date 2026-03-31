
"""
NeoBERT Modal Multi Pretraining Launcher
---------------------------------------

Sweep Variable Behavior:
-----------------------
- The SWEEP_KEYS list defines which config variables can be swept over (i.e., run multiple jobs with different values).
- For each SWEEP_KEY, the script checks for sweep values in this order:
    1. **CLI**: If a comma-separated value is provided in the CLI (e.g., model.loss.cost_based_loss_alpha_end=4e-8,4e-7,4e-6), it is treated as a sweep over those values.
    2. **YAML**: If not provided in the CLI, the script checks the YAML config. If the value is a YAML list (e.g., [4e-8, 4e-7, 4e-6]), it is treated as a sweep.
    3. If neither, the variable is treated as a single value.
- For each sweep variable, each run will receive a single value (never a list).

How to Specify Sweeps:
----------------------
- **From CLI:** Use comma-separated values (no brackets):
    model.loss.cost_based_loss_alpha_end=4e-8,4e-7,4e-6
- **From YAML:** Use a standard YAML list:
    model:
      loss:
        cost_based_loss_alpha_end: [4e-8, 4e-7, 4e-6]
- Do NOT use brackets or comma-separated strings in YAML (e.g., cost_based_loss_alpha_end: 4e-8,4e-7,4e-6) as this will be interpreted as a string, not a list.

The script will automatically detect and sweep over values from either source, launching one job per combination of sweep values.
"""
#Script for launching NeoBERT pretraining jobs on Modal with support for multiple model types and hyperparameter grid search.
#Handles CLI argument parsing, model type selection, and job submission to Modal cloud infrastructure.
import numpy as np
import sys
import modal
import itertools
from omegaconf import OmegaConf
from hydra import initialize, compose
from neobert.modal_runner import app, run_pretrain

# ------------------------------------------------
# CLI argument parsing
# ------------------------------------------------
def get_arg_value(key, cli_arguments):
    for arg_index, arg in enumerate(cli_arguments):  # skip script name
        if arg.startswith(f"{key}="):
            model_type_str = arg.split("=", 1)[1].strip('"').strip("'")
            cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index + 1 :]
            if model_type_str in ["mop", "hetero_moe", "homo_moe"]:
                return model_type_str, cli_overrides
            else:
                raise ValueError(
                    "Error: invalid model type. Must be 'mop', 'hetero_moe', or 'homo_moe'"
                )
    raise ValueError(
        "Error: no model_type given. Must be 'mop', 'hetero_moe', or 'homo_moe'"
    )


# ------------------------------------------------

# ------------------------------------------------
# Generalized hyperparameter grid search
# ------------------------------------------------

# Define sweepable variable keys (values come from config/overrides)
SWEEP_KEYS = [
    "model.loss.cost_based_loss_alpha_start",
    "model.loss.cost_based_loss_alpha_end",
    "model.loss.alpha_scaling",
    "model.loss.cost_based_loss_schedule_tokens",
    "model.expert_cost_exponent",
]

def parse_cli_sweep(cli_overrides, sweep_keys):
    """Return a dict: key -> list of values (if comma-separated in CLI), else None."""
    sweep_cli = {}
    for key in sweep_keys:
        for i, arg in enumerate(cli_overrides):
            if arg.startswith(f"{key}="):
                val = arg.split("=", 1)[1].strip()
                # Remove brackets if present (Hydra style)
                if val.startswith("[") and val.endswith("]"):
                    val = val[1:-1]
                # Split on comma if multiple values
                if "," in val:
                    vals = [v.strip() for v in val.split(",")]
                    # Try to convert to float if possible
                    def try_num(x):
                        try:
                            return float(x)
                        except Exception:
                            return x
                    vals = [try_num(v) for v in vals]
                    sweep_cli[key] = vals
                else:
                    # Try to convert to float if possible
                    try:
                        sweep_cli[key] = [float(val)]
                    except Exception:
                        sweep_cli[key] = [val]
                break
        else:
            sweep_cli[key] = None
    return sweep_cli

def run_many(cli_overrides_base, model_type):
    # Parse CLI for comma-separated sweep values for SWEEP_KEYS
    sweep_cli = parse_cli_sweep(cli_overrides_base, SWEEP_KEYS)
    # Remove any sweep keys from cli_overrides_base (will be added per run)
    filtered_cli = [
        arg for arg in cli_overrides_base
        if not any(arg.startswith(f"{k}=") for k in SWEEP_KEYS)
    ]
    # Load config with Hydra/OmegaConf to get sweep values from YAML/CLI
    with initialize(config_path="../../conf", version_base=None):
        config_name = f"pretraining_{model_type}"
        cfg = compose(config_name=config_name, overrides=filtered_cli)
        value_lists = []
        for key in SWEEP_KEYS:
            if sweep_cli[key] is not None:
                value_lists.append(sweep_cli[key])
            else:
                try:
                    val = OmegaConf.select(cfg, key)
                except Exception:
                    val = None
                if val is None:
                    value_lists.append([None])
                elif isinstance(val, list):
                    value_lists.append(val)
                else:
                    value_lists.append([val])
    futures = []
    with modal.enable_output(), app.run(detach=True):
        for values in itertools.product(*value_lists):
            # Only include override if value is not None
            overrides = filtered_cli + [
                f"{k}={v}" for k, v in zip(SWEEP_KEYS, values) if v is not None
            ]
            print(
                "Launching run: " + ", ".join(f"{k}={v}" for k, v in zip(SWEEP_KEYS, values) if v is not None)
            )
            fut = run_pretrain.spawn(
                overrides=overrides, model_type=model_type
            )
            futures.append(fut)
        results = [f.get() for f in futures]
    return results


# ------------------------------------------------
# Entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    cli_arguments = sys.argv[1:]  # exclude script name
    model_type, cli_overrides_base = get_arg_value("model_type", cli_arguments)

    print(f"Running hyperparameter sweep for model_type={model_type}")
    run_many(cli_overrides_base, model_type)
