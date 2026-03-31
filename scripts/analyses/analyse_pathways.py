"""
Script to launch NeoBERT token routing pathway analysis via Modal or locally.
Handles config loading and remote execution for pathway visualization.
"""
import sys

cli_arguments = sys.argv[1:]



from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "predictor"
    cfg_predictor = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_predictor.modal.run_on_modal
cli_overrides = cli_arguments


def pathways_analysis_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app, run_pathways_analysis

    with modal.enable_output(), app.run(detach=True):
        run_pathways_analysis.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        pathways_analysis_modal(cli_overrides)
