import sys

modal = True
cli_overrides = sys.argv[1:]


def predictor_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app,run_predictor
    with modal.enable_output(), app.run(detach=True):
        run_predictor.remote(overrides=cli_overrides)

if __name__ == "__main__":
    if modal == True:
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







