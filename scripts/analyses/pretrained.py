import sys


modal = True


def analyse_modal() -> None:
    import modal
    from neobert.modal_runner import app, run_analyses_pretrained

    with modal.enable_output(), app.run(detach=True):
        run_analyses_pretrained.remote()


if __name__ == "__main__":
    if modal == True:
        print("Running in Modal environment")
        analyse_modal()
