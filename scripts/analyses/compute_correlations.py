import sys

cli_arguments = sys.argv[1:]


def get_modal_flag(cli_arguments=cli_arguments):
    for arg_index, arg in enumerate(cli_arguments):
        if arg.startswith("modal="):
            modal_str = arg.split("=", 1)[1].strip('"').strip("'").lower()
            cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index + 1 :]
            if modal_str in ["true", "1", "yes"]:
                return True, cli_overrides
            elif modal_str in ["false", "0", "no"]:
                return False, cli_overrides
            else:
                print("Warning: invalid modal flag value. Must be 'true' or 'false'. Defaulting to True.")
                return True, cli_overrides
    return True, cli_arguments  # default to True if not specified


modal, cli_overrides = get_modal_flag(cli_arguments)


def correlations_modal() -> None:
    import modal
    from neobert.modal_runner import app, run_correlations

    with modal.enable_output(), app.run(detach=True):
        run_correlations.remote()


if __name__ == "__main__":
    if modal:
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
