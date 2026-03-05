import sys

modal = True
cli_overrides = sys.argv[1:]

def pretrained_model_tester_modal() -> None:
    import modal
    from neobert.modal_runner import app,run_pretrained_model_tester
    with modal.enable_output(), app.run(detach=True):
        run_pretrained_model_tester.remote(overrides=cli_overrides)
        
if __name__ == "__main__":
    if modal == True:
        print("Running in Modal environment")
        pretrained_model_tester_modal()







