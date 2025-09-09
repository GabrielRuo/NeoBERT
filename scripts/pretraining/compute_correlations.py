import sys

modal = True

def correlations_modal() -> None:
    import modal
    from neobert.modal_runner import app,run_correlations
    with modal.enable_output(), app.run(detach=True):
        run_correlations.remote()

if __name__ == "__main__":
    if modal == True:
        print("Running in Modal environment")
        correlations_modal()
    else:
        print("Running in local environment")
        from neobert.predictor import predictor
        from hydra import initialize, compose
        with initialize(config_path="../../conf", version_base=None):
            config_name = "predictor"
            cfg_predictor = compose(config_name=config_name, overrides=cli_overrides)
            predictor(cfg_predictor)







