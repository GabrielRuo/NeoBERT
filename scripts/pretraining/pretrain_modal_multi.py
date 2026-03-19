import sys
import modal
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
# Hyperparameter grids
# ------------------------------------------------
# alpha_start_list = [5e-10]
# alpha_end_list = [2e-7,1e-7,5e-8, 2e-8,1e-8]
# schedule_list = [2500000, 3000000, 3500000]
# cost_exponent_list = [1,1.5,2]

# alpha_start_list = [5e-10]
# alpha_end_list = [1e-9,2e-8,3e-8,4e-8,7e-8,2e-7,3e-7,4e-7,5e-7,6e-7,7e-7,1e-6,5e-6,1e-5]
# alpha_scaling_list = [0.0,1.0,5.0]
# schedule_list = [2500000]
# cost_exponent_list = [2]
import numpy as np

alpha_start_list = [1e-10]  # [5e-10]
alpha_end_list = list(np.array([4e-7, 6e-7, 8e-7, 1.3e-6, 1.6e-6, 3e-6]))
# alpha_end_list = list(np.array([0.0,5e-9,8e-9,2e-8,4e-8,7e-8,9e-8,2e-7,4e-7,6e-7,8e-7,1e-6,5e-6,1e-5])/5)
alpha_scaling_list = [0.0]
schedule_list = [2500000]
cost_exponent_list = [2]

# alpha_start_list = [5e-10]
# alpha_end_list = [4e-8]
# alpha_scaling_list = [0.0]
# schedule_list = [2500000]
# cost_exponent_list = [2]


# ------------------------------------------------
# Launch many jobs on Modal
# ------------------------------------------------
def run_many(cli_overrides_base, model_type):
    futures = []
    with modal.enable_output(), app.run(detach=True):
        for alpha_start in alpha_start_list:
            for alpha_end in alpha_end_list:
                for schedule in schedule_list:
                    for cost_exponent in cost_exponent_list:
                        for alpha_scaling in alpha_scaling_list:
                            overrides = cli_overrides_base + [
                                f"model.loss.cost_based_loss_alpha_start={alpha_start}",
                                f"model.loss.cost_based_loss_alpha_end={alpha_end}",
                                f"model.loss.cost_based_loss_schedule_tokens={schedule}",
                                f"model.expert_cost_exponent={cost_exponent}",
                                f"model.loss.alpha_scaling={alpha_scaling}",
                            ]
                            print(
                                f"Launching run: "
                                f"alpha_start={alpha_start}, alpha_end={alpha_end}, "
                                f"schedule={schedule}, cost_exp={cost_exponent}, alpha_scaling={alpha_scaling}"
                            )
                            # non-blocking launch
                            fut = run_pretrain.spawn(
                                overrides=overrides, model_type=model_type
                            )
                            futures.append(fut)

        # (Optional) wait for results
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
