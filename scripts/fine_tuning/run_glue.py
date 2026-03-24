import sys

cli_arguments = sys.argv[1:]



from hydra import initialize, compose
with initialize(config_path="../../conf", version_base=None):
    config_name = "glue_mop"
    cfg = compose(config_name=config_name, overrides=cli_arguments)
modal = cfg.modal.run_on_modal
cli_overrides = cli_arguments

# def get_arg_value(cli_arguments=cli_arguments):
#     for arg_index,arg in enumerate(cli_arguments):  # skip the script name
#         cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index+1:]
#     return cli_overrides


# cli_overrides = get_arg_value(cli_arguments)

print(cli_overrides)


def pipeline_modal(cli_overrides: list[str]) -> None:
    import modal
    from neobert.modal_runner import app, run_glue

    with modal.enable_output(), app.run(detach=True):
        run_glue.remote(overrides=cli_overrides)


if __name__ == "__main__":
    if modal:
        print("Running in Modal environment")
        pipeline_modal(cli_overrides)
    else:
        print("Running in local environment")
        from neobert.glue import trainer
        from hydra import initialize, compose

        with initialize(config_path="../../conf", version_base=None):
            config_name = "glue_mop"
            cfg = compose(config_name=config_name, overrides=cli_overrides)
            trainer(cfg)
