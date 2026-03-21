import sys
from dotenv import load_dotenv

# Load secrets from .env file
# ------------------------------
# This reads environment variables from a local .env file.
# Make sure your .env file contains:
#   HF_TOKEN=your_huggingface_token_here
#   WANDB_API_KEY=your_wandb_api_key_here
load_dotenv()

cli_arguments = sys.argv[1:]


def get_modal_flag(cli_arguments=cli_arguments):
    for arg_index, arg in enumerate(cli_arguments):
        if arg.startswith("modal="):
            modal_str = arg.split("=", 1)[1].strip('"').strip("'").lower()
            cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index + 1 :]
            if modal_str in ["true", "1", "yes"]:
                return True, cli_overrides
            elif modal_str in ["false", "0", "no"]:
                return False, cli_overrides
            else:
                print("Warning: invalid modal flag value. Must be 'true' or 'false'. Defaulting to False.")
                return False, cli_overrides
    return False, cli_arguments  # default to False if not specified


modal, cli_overrides = get_modal_flag(cli_arguments)


def predictor_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app, run_predictor

    with modal.enable_output(), app.run(detach=True):
        run_predictor.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        predictor_modal(cli_overrides)
    else:
        print("Running in local environment")
        from neobert.predictor import predictor
        from hydra import initialize, compose

        with initialize(config_path="../../conf", version_base=None):
            config_name = "predictor"
            cfg_predictor = compose(config_name=config_name, overrides=cli_overrides)
            predictor(cfg_predictor)
