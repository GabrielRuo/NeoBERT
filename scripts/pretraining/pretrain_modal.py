import sys
from dotenv import load_dotenv

# Load secrets from .env file
# ------------------------------
# This reads environment variables from a local .env file.
# Make sure your .env file contains:
#   HF_TOKEN=your_huggingface_token_here
#   WANDB_API_KEY=your_wandb_api_key_here
#If using docker container, the env variables should  already be loaded
#into the container
load_dotenv()

cli_arguments = sys.argv[1:]


def get_arg_value(key, cli_arguments=cli_arguments):
    for arg_index, arg in enumerate(cli_arguments):  # skip the script name
        if arg.startswith(f"{key}="):
            model_type_str = arg.split("=", 1)[1].strip('"').strip("'")
            cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index + 1 :]
            if model_type_str in ["mop", "hetero_moe", "homo_moe", "neobert_original"]:
                return model_type_str, cli_overrides
            else:
                print(
                    "Error: invalid model type. Must be 'mop', 'hetero_moe', 'homo_moe', or 'neobert_original'"
                )
                return None
    print(
        "Error: no model_type given. Must be 'mop', 'hetero_moe', 'homo_moe', or 'neobert_original'"
    )
    return None



model_type, cli_overrides = get_arg_value("model_type", cli_arguments)
from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "pretraining_" + model_type
    cfg = compose(config_name=config_name, overrides=cli_overrides)
modal = cfg.modal.run_on_modal


def pipeline_modal(cli_overrides: list[str], model_type) -> None:
    import modal
    from neobert.modal_runner import app, run_pretrain

    with modal.enable_output(), app.run(detach=True):
        run_pretrain.remote(overrides=cli_overrides, model_type=model_type)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        pipeline_modal(cli_overrides, model_type)
    else:
        print("Running in local environment")
        from neobert.pretraining import trainer
        from hydra import initialize, compose

        with initialize(config_path="../../conf", version_base=None):
            config_name = "pretraining_" + model_type
            cfg = compose(config_name=config_name, overrides=cli_overrides)
            trainer(cfg)
