import sys

modal = True
cli_arguments= sys.argv[1:]

def get_arg_value(key, cli_arguments=cli_arguments):
    for arg_index,arg in enumerate(cli_arguments):  # skip the script name
        if arg.startswith(f"{key}="):
            model_type_str =  arg.split("=", 1)[1].strip('"').strip("'")
            cli_overrides = cli_arguments[:arg_index] + cli_arguments[arg_index+1:]
            if model_type_str in ["mop", "hetero_moe", "homo_moe"]:
                return model_type_str, cli_overrides
            else:
                print("Error: invalid model type. Must be 'mop', 'hetero_moe', or 'homo_moe'")
                return None
    print("Error: no model_type given. Must be 'mop', 'hetero_moe', or 'homo_moe'")
    return None

model_type, cli_overrides = get_arg_value("model_type", cli_arguments)


def pipeline_modal(cli_overrides: list[str], model_type) -> None:
    import modal
    from neobert.modal_runner import app,run_pretrain
    with modal.enable_output(), app.run(detach=True):
        run_pretrain.remote(overrides=cli_overrides, model_type=model_type)

if __name__ == "__main__":
    if modal == True:
        print("Running in Modal environment")
        pipeline_modal(cli_overrides, model_type)
    else:
        print("Running in local environment")
        from neobert.pretraining import trainer
        from hydra import initialize, compose
        with initialize(config_path="../../conf", version_base=None):
            config_name = "pretraining_" + model_type
            cfg = compose(config_name=config_name, overrides=cli_overrides)
            trainer(cfg)







