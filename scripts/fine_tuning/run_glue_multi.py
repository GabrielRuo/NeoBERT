"""
NeoBERT Modal Multi GLUE Launcher
---------------------------------

Sweep Variable Behavior:
-----------------------
- The SWEEP_KEYS list defines which config variables can be swept over.
- For each SWEEP_KEY, values are resolved in this order:
    1. CLI comma-separated values (e.g. key=1e-4,1e-5)
    2. YAML list values from the Hydra config
    3. Single scalar value
- Each launched run always receives a single value per sweep key.
"""

import itertools
import sys

import modal
from hydra import compose, initialize
from omegaconf import OmegaConf

from neobert.modal_runner import app, run_glue


# ------------------------------------------------
# Generalized hyperparameter grid search
# ------------------------------------------------

# Define sweepable variable keys (values come from config/overrides)
SWEEP_KEYS = [
    "model.loss.cost_based_loss_alpha_end",
    "model.loss.load_balancing_loss_coeff",
]


def parse_cli_sweep(cli_overrides, sweep_keys):
    """Return a dict: key -> list of values (if supplied via CLI), else None."""
    sweep_cli = {}
    for key in sweep_keys:
        for arg in cli_overrides:
            if arg.startswith(f"{key}="):
                val = arg.split("=", 1)[1].strip()
                if val.startswith("[") and val.endswith("]"):
                    val = val[1:-1]

                if "," in val:
                    vals = [v.strip() for v in val.split(",")]

                    def try_num(x):
                        try:
                            return float(x)
                        except Exception:
                            return x

                    sweep_cli[key] = [try_num(v) for v in vals]
                else:
                    try:
                        sweep_cli[key] = [float(val)]
                    except Exception:
                        sweep_cli[key] = [val]
                break
        else:
            sweep_cli[key] = None
    return sweep_cli


def run_many(cli_overrides_base):
    sweep_cli = parse_cli_sweep(cli_overrides_base, SWEEP_KEYS)

    # Remove sweep keys from base overrides so each run gets one scalar per key.
    filtered_cli = [
        arg
        for arg in cli_overrides_base
        if not any(arg.startswith(f"{k}=") for k in SWEEP_KEYS)
    ]

    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="glue_mop", overrides=filtered_cli)
        value_lists = []

        for key in SWEEP_KEYS:
            if sweep_cli[key] is not None:
                value_lists.append(sweep_cli[key])
                continue

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
            overrides = filtered_cli + [
                f"{k}={v}" for k, v in zip(SWEEP_KEYS, values) if v is not None
            ]
            print(
                "Launching run: "
                + ", ".join(
                    f"{k}={v}" for k, v in zip(SWEEP_KEYS, values) if v is not None
                )
            )
            fut = run_glue.spawn(overrides=overrides)
            futures.append(fut)

        results = [f.get() for f in futures]
    return results


# ------------------------------------------------
# Entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    cli_arguments = sys.argv[1:]  # exclude script name
    print("Running hyperparameter sweep")
    run_many(cli_arguments)
