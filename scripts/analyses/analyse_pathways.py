import sys

modal = True
cli_overrides = sys.argv[1:]


def pathways_analysis_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app,run_pathways_analysis
    with modal.enable_output(), app.run(detach=True):
        run_pathways_analysis.remote(overrides=cli_overrides)

if __name__ == "__main__":
    if modal == True:
        print("Running in Modal environment")
        pathways_analysis_modal(cli_overrides)







