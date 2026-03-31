import sys

cli_arguments = sys.argv[1:]



from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "difficulty"
    cfg_dif = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_dif.modal.run_on_modal
cli_overrides = cli_arguments


def difficulty_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app, run_difficulty

    with modal.enable_output(), app.run(detach=True):
        run_difficulty.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        difficulty_modal(cli_overrides)
    else:
        print("Running in local environment")

        from neobert.difficulty import measure_difficulty
        from hydra import initialize, compose

        with initialize(config_path="../../conf", version_base=None):
            config_name = "difficulty"
            cfg_dif = compose(config_name=config_name, overrides=cli_overrides)
            measure_difficulty(cfg_dif)
