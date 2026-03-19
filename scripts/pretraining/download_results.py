import os
from pathlib import Path
import modal
import argparse
from typing import Optional
from typing import Union
import shutil
import sys

# Local target directory
local_logs_dir = Path(__file__).parent.parent / "src" / "neobert" / "runs" / "logs"
local_logs_dir.mkdir(parents=True, exist_ok=True)

# def _process_entry(entry,volume,remote_dir,local_dir, checkpoint, delete_in_modal):
#     remote_path = entry.path
#     local_path = local_dir / os.path.relpath(remote_path, remote_dir)
#     if entry.type.name == "DIRECTORY":
#         print(f"Creating local directory: {local_path}")
#         local_path.mkdir(parents=True, exist_ok=True)
#         download_directory(volume, remote_path, local_path, checkpoint, delete_in_modal)
#     elif entry.type.name == "FILE":
#         if local_path.exists():
#             print(f"Skipping already existing file: {local_path}")
#             continue
#         print(f"Downloading {remote_path} to {local_path}")
#         local_path.parent.mkdir(parents=True, exist_ok=True)
#         file_bytes_gen = volume.read_file(remote_path)
#         file_bytes = b''.join(file_bytes_gen)
#         with open(local_path, "wb") as dst:
#             dst.write(file_bytes)
#     else:
#         print(f"Skipping unknown entry type: {remote_path}")

# def delete_remote_directory(volume, remote_dir_str):
#     """
#     Recursively delete a directory and its contents from the Modal volume.
#     """
#     entries = volume.listdir(remote_dir_str)
#     for entry in entries:
#         if entry.type.name == "DIRECTORY":
#             delete_remote_directory(volume, entry.path)
#         elif entry.type.name == "FILE":
#             print(f"Deleting remote file: {entry.path}")
#             volume.remove_file(entry.path)
#     print(f"Deleting remote directory: {remote_dir_str}")
#     volume.remove_dir(remote_dir_str)


def download_directory(
    volume,
    remote_dir_str,
    local_dir,
    checkpoint: str = "All",
    delete_in_modal: bool = False,
):

    remote_dir = Path(remote_dir_str)

    # print(volume.listdir(model_checkpoints_dir_str))
    # print(type(volume.listdir(remote_dir_str)))
    # print(type(volume.listdir(model_checkpoints_dir_str)[0]))

    if checkpoint != "All":
        # Download a specific checkpoint
        target = remote_dir_str + "/model_checkpoints/" + checkpoint
        target_str = str(target).replace("\\", "/")

        entry = volume.listdir(target_str)[0]  # just get the state dict
        # local_dir = local_dir/checkpoint

        remote_path = entry.path
        local_path = local_dir / os.path.relpath(remote_path, remote_dir)
        if local_path.exists():
            print(f"Skipping already existing file: {local_path}")
            return
        print(f"Downloading {remote_path} to {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_bytes_gen = volume.read_file(remote_path)
        file_bytes = b"".join(file_bytes_gen)
        with open(local_path, "wb") as dst:
            dst.write(file_bytes)
        # add the config file
        config_path_str = remote_dir_str + "/config.yaml"
        config_path_str = config_path_str.replace("\\", "/")
        config_file_bytes_gen = volume.read_file(config_path_str)
        config_file_bytes = b"".join(config_file_bytes_gen)
        with open(local_dir / "config.yaml", "wb") as dst:
            dst.write(config_file_bytes)

    else:
        # Download all checkpoints
        checkpoint_entries = volume.listdir(remote_dir_str + "/model_checkpoints")
        for entry in checkpoint_entries:
            checkpoint_value = entry.path.split("/")[-1]
            # local_dir = local_dir / checkpoint_value
            remote_path = entry.path
            local_path = local_dir / os.path.relpath(entry.path, remote_dir)
            if entry.type.name == "DIRECTORY":
                print(f"Creating local directory: {local_path}")
                local_path.mkdir(parents=True, exist_ok=True)
                download_directory(
                    volume, remote_dir_str, local_dir, checkpoint_value, delete_in_modal
                )
            elif entry.type.name == "FILE":
                if local_path.exists():
                    print(f"Skipping already existing file: {local_path}")
                    continue
                print(f"Downloading {remote_path} to {local_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                file_bytes_gen = volume.read_file(remote_path)
                file_bytes = b"".join(file_bytes_gen)
                with open(local_path, "wb") as dst:
                    dst.write(file_bytes)
            else:
                print(f"Skipping unknown entry type: {remote_path}")


def download_runs_logs(
    base_path: Union[list[str], None] = None,
    checkpoints: Union[str, list[str]] = ["1000,2000"],
    delete_in_modal: bool = False,
):
    print("Starting download of runs logs from Modal...")
    # Connect to the Modal volume
    runs_volume = modal.Volume.from_name("neobert-runs", create_if_missing=False)
    # Determine remote and local directories

    def build_dirs(base_path):
        remote_logs_dir = os.path.join("logs", os.path.join("checkpoints", base_path))
        remote_logs_dir_str = "logs/checkpoints" + "/" + base_path
        remote_logs_dir_str = remote_logs_dir_str.replace("\\", "/")
        # print(f"Remote logs directory: {remote_logs_dir_str}")
        remote_logs_dir = Path(remote_logs_dir_str)
        # Save under logs/relative_path locally
        # local_dir = os.path.join(local_logs_dir, base_path,"model_checkpoints")# we do not add the value of the chckpoint yet. It will be added in download_directory
        # print(f"Downloading logs from remote directory: {remote_logs_dir} to {local_dir}")
        local_dir = local_logs_dir / Path(remote_logs_dir).relative_to("logs")
        return remote_logs_dir_str, local_dir

    print(f"Base path type: {type(base_path)}")

    if base_path == None:
        remote_logs_dir_str = "logs"
        local_dir = local_logs_dir
        if checkpoints == "All":
            download_directory(
                runs_volume, remote_logs_dir_str, local_dir, delete_in_modal
            )
        elif checkpoints == "None":
            if delete_in_modal:
                print(f"Deleting remote directory: {remote_logs_dir_str}")
                delete_in_volume.remote(remote_dir=remote_logs_dir_str)
        else:
            for checkpoint in checkpoints:
                remote_logs_dir_str, local_dir = build_dirs(base_path)
                download_directory(
                    runs_volume,
                    remote_logs_dir_str,
                    local_dir,
                    checkpoint,
                    delete_in_modal,
                )
        if delete_in_modal and checkpoints != "None":
            print(f"Deleting remote directory: {remote_logs_dir_str}")
            delete_in_volume.remote(remote_dir=remote_logs_dir_str)
    elif type(base_path) == list:
        print("Downloading specified base paths...")
        if base_path[0] == "":
            print("Empty base path provided. Exiting.")
            return
        for path in base_path:
            # download the config file
            remote_logs_dir_str, local_dir = build_dirs(path)
            glue_path = remote_logs_dir_str + "/glue"
            print("glue path:", glue_path)
            if delete_glue_in_modal:
                print(f"Deleting remote directory: {glue_path}")
                delete_in_volume.remote(remote_dir=glue_path)

            if checkpoints == "All":
                download_directory(
                    runs_volume,
                    remote_logs_dir_str,
                    local_dir,
                    checkpoints,
                    delete_in_modal,
                )

            elif checkpoints == "None":
                print(f"No checkpoints specified for download for base path: {path}.")
                if delete_in_modal:
                    print(f"Deleting remote directory: {remote_logs_dir_str}")
                    delete_in_volume.remote(remote_dir=remote_logs_dir_str)
            else:
                for checkpoint in checkpoints:
                    print(checkpoint)
                    try:
                        download_directory(
                            runs_volume,
                            remote_logs_dir_str,
                            local_dir,
                            checkpoint,
                            delete_in_modal,
                        )
                    except Exception as e:
                        print(
                            f"error downloading checkpoint {checkpoint}: {e}. Moving to next checkpoint."
                        )
                        continue

        if delete_in_modal and checkpoints != "None":
            print(f"Deleting remote directory: {remote_logs_dir_str}")
            delete_in_volume.remote(remote_dir=remote_logs_dir_str)


if __name__ == "__main__":
    import modal
    from neobert.modal_runner import app, delete_in_volume

    # Parse key=value arguments from sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    args = parser.parse_args()

    # Defaults
    base_path = None
    checkpoints = "All"
    delete_in_modal = False
    delete_glue_in_modal = False

    # Parse key=value pairs
    for arg in args.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "base_path":
                base_path = value
            elif key == "checkpoints":
                checkpoints = value
            elif key == "delete_in_modal":
                delete_in_modal = value.lower() in ("true", "1", "yes")
            elif key == "delete_glue_in_modal":
                delete_glue_in_modal = value.lower() in ("true", "1", "yes")
            else:
                print(f"Unknown argument: {key}")

    if base_path is None:
        # log a question asking whether you want to download all runs from modal?
        response = input(
            "No base_path provided. Do you want to download all runs from Modal? (y/n): "
        )
        if response.lower() != "n":
            print("Exiting without downloading.")
            sys.exit(0)
    else:
        base_path = base_path.split(",")

    if delete_in_modal:
        response = input("Do you want to delete the modal runs? CAREFUL (y/n): ")
        if response.lower() == "n":
            print("Exiting without deleting.")
            sys.exit(0)

    if checkpoints != "All" and checkpoints != "None":
        checkpoints = checkpoints.split(",")
    with modal.enable_output(), app.run(detach=True):
        download_runs_logs(
            base_path=base_path,
            checkpoints=checkpoints,
            delete_in_modal=delete_in_modal,
        )

    # currently I have extraceted the loop from download_runs_logs which throws an error. Will have to correct that

# maintenant: delete le modèle sur modal
# sauvegarder la config du  run
