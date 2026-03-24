import sys

cli_arguments = sys.argv[1:]



from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "tester"
    cfg_tester = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_tester.modal.run_on_modal


def pretrained_model_tester_modal() -> None:
    import modal
    from neobert.modal_runner import app, run_pretrained_model_tester

    with modal.enable_output(), app.run(detach=True):
        run_pretrained_model_tester.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        pretrained_model_tester_modal()
