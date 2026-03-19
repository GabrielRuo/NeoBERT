import os
import shlex
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


pytestmark = [pytest.mark.local, pytest.mark.e2e, pytest.mark.slow]


RUN_OFFLINE_E2E = os.getenv("RUN_OFFLINE_E2E", "0") == "1"


def _has_override(overrides, key):
    prefix = f"{key}="
    return any(item.lstrip("+").startswith(prefix) for item in overrides)


@pytest.mark.skipif(
    not RUN_OFFLINE_E2E,
    reason="Set RUN_OFFLINE_E2E=1 to run the full offline E2E pipeline",
)
def test_predict_routing_full_pipeline_offline_local_assets():
    """Run predict_routing with local-only model + datasets and no network access.

    This test executes the real script entrypoint and is intentionally opt-in
    because it depends on pre-existing local artifacts.
    """

    repo_root = Path(__file__).resolve().parents[1]

    if importlib.util.find_spec("dotenv") is None:
        pytest.skip(
            "python-dotenv is required by scripts/analyses/predict_routing.py. "
            "Install it to run this offline E2E test."
        )

    script_path = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_SCRIPT",
        "scripts/analyses/predict_routing.py",
    )

    timeout_s = int(os.getenv("PREDICT_ROUTING_OFFLINE_E2E_TIMEOUT", "3600"))
    extra_overrides = shlex.split(os.getenv("PREDICT_ROUTING_OFFLINE_E2E_OVERRIDES", ""))

    default_base_path = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_BASE_PATH",
        "/runs/logs/checkpoints/mop_2025-12-02_16-36-59/",
    )
    default_checkpoint = os.getenv("PREDICT_ROUTING_OFFLINE_E2E_CHECKPOINT", "40000")
    default_train_dataset = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_TRAIN_DATASET_PATH",
        "/data/.pathways_cache/jeankaddourminipiletrain_100",
    )
    default_test_dataset = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_TEST_DATASET_PATH",
        default_train_dataset,
    )

    expected_paths = {
        "saved model base path": Path(default_base_path),
        "saved model config": Path(default_base_path) / "config.yaml",
        "saved model state dict": (
            Path(default_base_path)
            / "model_checkpoints"
            / str(default_checkpoint)
            / "state_dict.pt"
        ),
        "train dataset": Path(default_train_dataset),
        "test dataset": Path(default_test_dataset),
    }

    missing = [f"{name}: {path}" for name, path in expected_paths.items() if not path.exists()]
    if missing:
        pytest.skip(
            "Offline E2E assets missing. Set PREDICT_ROUTING_OFFLINE_E2E_* env vars "
            f"to valid local paths. Missing: {', '.join(missing)}"
        )

    if not _has_override(extra_overrides, "saved_model.base_path"):
        extra_overrides.append(f"saved_model.base_path={default_base_path}")
    if not _has_override(extra_overrides, "saved_model.checkpoint"):
        extra_overrides.append(f"saved_model.checkpoint={default_checkpoint}")
    if not _has_override(extra_overrides, "dataset.train.path_to_disk"):
        extra_overrides.append(f"+dataset.train.path_to_disk={default_train_dataset}")
    if not _has_override(extra_overrides, "dataset.test.path_to_disk"):
        extra_overrides.append(f"+dataset.test.path_to_disk={default_test_dataset}")

    cmd = [
        sys.executable,
        script_path,
        "trainer.max_steps=1",
        "trainer.disable_tqdm=True",
        "trainer.test_after_training=False",
        "trainer.mixed_precision=no",
        "trainer.gradient_accumulation_steps=1",
        "dataloader.train.batch_size=2",
        "dataloader.train.num_workers=0",
        "dataloader.train.persistent_workers=False",
        "dataloader.test.batch_size=2",
        "dataloader.test.num_workers=0",
        "dataloader.test.persistent_workers=False",
        "val_interval=9999",
        "val_batches_per_eval=1",
        "wandb.mode=offline",
        "wandb.log_interval=1",
    ] + extra_overrides

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    assert result.returncode == 0, (
        "Offline E2E pipeline failed.\n"
        f"command: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )


@pytest.mark.skipif(
    not RUN_OFFLINE_E2E,
    reason="Set RUN_OFFLINE_E2E=1 to run the full offline E2E pipeline",
)
def test_pretrain_full_pipeline_offline_local_dataset(tmp_path):
    """Run pretrain.py end-to-end with a local synthetic dataset in offline mode."""

    from datasets import Dataset
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    script_path = os.getenv(
        "PRETRAIN_OFFLINE_E2E_SCRIPT",
        "scripts/pretraining/pretrain.py",
    )

    timeout_s = int(os.getenv("PRETRAIN_OFFLINE_E2E_TIMEOUT", "3600"))
    extra_overrides = shlex.split(os.getenv("PRETRAIN_OFFLINE_E2E_OVERRIDES", ""))

    tokenizer_name = os.getenv(
        "PRETRAIN_OFFLINE_E2E_TOKENIZER",
        "google-bert/bert-base-uncased",
    )
    try:
        AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception as exc:
        pytest.skip(
            "Offline pretrain E2E requires a locally cached tokenizer. "
            f"Missing tokenizer {tokenizer_name!r}: {exc}"
        )

    dataset_path = tmp_path / "offline_pretrain_ds"
    ds = Dataset.from_dict(
        {
            "input_ids": [
                [101, 2023, 2003, 1037, 7953, 7099, 102],
                [101, 1045, 2293, 2023, 3899, 3231, 102],
                [101, 1045, 2572, 2066, 2122, 5604, 102],
                [101, 1996, 3899, 2003, 7078, 102, 0],
            ]
        }
    )
    ds.save_to_disk(str(dataset_path))

    if not _has_override(extra_overrides, "dataset"):
        extra_overrides.append("dataset=minipile")
    if not _has_override(extra_overrides, "tokenizer"):
        extra_overrides.append("tokenizer=google_mini")

    cmd = [
        sys.executable,
        script_path,
        "--config-name",
        "pretraining_mop",
        "trainer.max_steps=1",
        "trainer.resume=False",
        "trainer.save_model=False",
        "trainer.disable_tqdm=True",
        "trainer.mixed_precision=no",
        "trainer.gradient_accumulation_steps=1",
        "dataloader.train.batch_size=1",
        "dataloader.train.num_workers=0",
        "dataloader.train.persistent_workers=False",
        "dataloader.train.pin_memory=False",
        f"trainer.dir={tmp_path.as_posix()}",
        f"dataset.train.path_to_disk={dataset_path.as_posix()}",
        "wandb.mode=offline",
        "wandb.log_interval=1",
        "model.hidden_size=8",
        "model.num_hidden_layers=2",
        "model.num_attention_heads=2",
        "model.expert_sizes='0,24,48'",
        "model.num_experts_per_tok_training=2",
        "model.num_experts_per_tok_inference=1",
    ] + extra_overrides

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    env["TORCH_COMPILE_DISABLE"] = "1"

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    assert result.returncode == 0, (
        "Offline pretrain E2E pipeline failed.\n"
        f"command: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
