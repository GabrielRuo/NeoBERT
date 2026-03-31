import sys

cli_arguments = sys.argv[1:]



from hydra import initialize, compose
"""
Script to launch NeoBERT pretrained model analysis via Modal or locally.
Handles config loading and remote execution for pretrained model analysis.
"""
import sys

cli_arguments = sys.argv[1:]


from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "tester"
    cfg_tester = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg_tester.modal.run_on_modal


def analyse_modal() -> None:
    import modal
    from neobert.modal_runner import app, run_analyses_pretrained

    with modal.enable_output(), app.run(detach=True):
        run_analyses_pretrained.remote()


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        analyse_modal()
