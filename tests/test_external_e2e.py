import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.external, pytest.mark.e2e, pytest.mark.slow]


RUN_EXTERNAL_E2E = os.getenv("RUN_EXTERNAL_E2E", "0") == "1"


def _has_override(overrides, key):
    prefix = f"{key}="
    return any(item.startswith(prefix) for item in overrides)


@pytest.mark.skipif(
    not RUN_EXTERNAL_E2E,
    reason="Set RUN_EXTERNAL_E2E=1 to run the full external E2E pipeline",
)
def test_predict_routing_full_pipeline_external():
    """Run the real predict_routing entrypoint with online services enabled.

    This test is intentionally opt-in because it is expensive and depends on
    external systems (network, Hugging Face, W&B, and optionally Modal settings).
    """

    repo_root = Path(__file__).resolve().parents[1]
    script_path = os.getenv(
        "PREDICT_ROUTING_E2E_SCRIPT", "scripts/analyses/predict_routing.py"
    )

    timeout_s = int(os.getenv("PREDICT_ROUTING_E2E_TIMEOUT", "3600"))
    extra_overrides = shlex.split(os.getenv("PREDICT_ROUTING_E2E_OVERRIDES", ""))

    # Keep E2E test parameters local to the test so changes in predictor.yaml
    # do not silently change external E2E behavior.
    default_base_path = os.getenv(
        "PREDICT_ROUTING_E2E_BASE_PATH",
        "/runs/logs/checkpoints/mop_2025-12-02_16-36-59/",
    )
    default_checkpoint = os.getenv("PREDICT_ROUTING_E2E_CHECKPOINT", "40000")

    if not _has_override(extra_overrides, "saved_model.base_path"):
        extra_overrides.append(f"saved_model.base_path={default_base_path}")
    if not _has_override(extra_overrides, "saved_model.checkpoint"):
        extra_overrides.append(f"saved_model.checkpoint={default_checkpoint}")

    cmd = [
        sys.executable,
        script_path,
        "trainer.max_steps=1",
        "trainer.disable_tqdm=True",
        "trainer.test_after_training=False",
        "val_interval=9999",
        "val_batches_per_eval=1",
        "wandb.mode=online",
        "wandb.log_interval=1",
    ] + extra_overrides

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    assert result.returncode == 0, (
        "External E2E pipeline failed.\n"
        f"command: {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
