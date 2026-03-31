import sys

cli_arguments = sys.argv[1:]


from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "pathways_visualiser"
    cfg_visualise_pathways = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_visualise_pathways.modal.run_on_modal
cli_overrides = cli_arguments


def visualise_pathways_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app, run_visualise_pathways

    with modal.enable_output(), app.run(detach=True):
        run_visualise_pathways.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        visualise_pathways_modal(cli_overrides)
