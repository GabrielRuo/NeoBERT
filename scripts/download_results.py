import os
from pathlib import Path
import modal

# Local target directory
local_logs_dir = Path(__file__).parent.parent / 'src' / 'neobert' / 'runs' / 'logs'
local_logs_dir.mkdir(parents=True, exist_ok=True)

def download_directory(volume, remote_dir, local_dir):
    entries = volume.listdir(remote_dir)
    for entry in entries:
        remote_path = entry.path
        local_path = local_dir / os.path.relpath(remote_path, remote_dir)
        if entry.type.name == "DIRECTORY":
            print(f"Creating local directory: {local_path}")
            local_path.mkdir(parents=True, exist_ok=True)
            download_directory(volume, remote_path, local_path)
        elif entry.type.name == "FILE":
            if local_path.exists():
                print(f"Skipping already existing file: {local_path}")
                continue
            print(f"Downloading {remote_path} to {local_path}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            file_bytes_gen = volume.read_file(remote_path)
            file_bytes = b''.join(file_bytes_gen)
            with open(local_path, "wb") as dst:
                dst.write(file_bytes)
        else:
            print(f"Skipping unknown entry type: {remote_path}")

def download_runs_logs():
    # Connect to the Modal volume
    runs_volume = modal.Volume.from_name("neobert-runs", create_if_missing=False)
    # List files in the logs directory
    remote_logs_dir = "logs"
    download_directory(runs_volume, remote_logs_dir, local_logs_dir)

if __name__ == "__main__":
    download_runs_logs()
