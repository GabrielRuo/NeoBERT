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


def move_runs_logs_from_local_to_modal(
    local_dir, base_path=base_path, checkpoints=checkpoints
):
    """
    Move runs and logs from local directory to modal volume.

    Args:
        local_dir (str): Path to the local directory containing runs and logs.
        base_path (str, optional): Base path in the modal volume. Defaults to "/neobert/runs/logs".
        checkpoints (Union[str, list], optional): Checkpoints to move. "All" to move all checkpoints. Defaults to "All".
    """
    if base_path is None:
        base_path = "/neobert/runs/logs"

    modal_volume_path = Path(base_path)
    modal_volume_path.mkdir(parents=True, exist_ok=True)

    for run_dir in os.listdir(local_dir):
        local_run_path = Path(local_dir) / run_dir
        modal_run_path = modal_volume_path / run_dir
        modal_run_path.mkdir(parents=True, exist_ok=True)

        if checkpoints == "All":
            for item in os.listdir(local_run_path):
                s = local_run_path / item
                d = modal_run_path / item
                if s.is_dir():
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        else:
            if isinstance(checkpoints, str):
                checkpoints = [checkpoints]
            for checkpoint in checkpoints:
                local_checkpoint_path = local_run_path / checkpoint
                modal_checkpoint_path = modal_run_path / checkpoint
                if local_checkpoint_path.exists():
                    if local_checkpoint_path.is_dir():
                        shutil.copytree(local_checkpoint_path, modal_checkpoint_path)
                    else:
                        shutil.copy2(local_checkpoint_path, modal_checkpoint_path)
                else:
                    print(f"Checkpoint {checkpoint} does not exist in {local_run_path}")


if __name__ == "__main__":
    import modal
    from neobert.modal_runner import app, move_to_volume

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
            else:
                print(f"Unknown argument: {key}")

    with modal.enable_output(), app.run(detach=True):
        move_runs_logs_from_local_to_modal(base_path=base_path, checkpoints=checkpoints)

    # currently I have extraceted the loop from download_runs_logs which throws an error. Will have to correct that

# maintenant: delete le modèle sur modal
# sauvegarder la config du  run
