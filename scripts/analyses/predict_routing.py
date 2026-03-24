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



from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "predictor"
    cfg_predictor = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_predictor.modal.run_on_modal
cli_overrides = cli_arguments


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
