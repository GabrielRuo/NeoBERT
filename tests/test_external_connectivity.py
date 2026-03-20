import os
import subprocess
import sys
import json
import urllib.request

import pytest


pytestmark = [pytest.mark.external]


RUN_EXTERNAL = os.getenv("RUN_EXTERNAL_TESTS", "0") == "1"


def _require_external():
    if not RUN_EXTERNAL:
        pytest.skip("Set RUN_EXTERNAL_TESTS=1 to run external connectivity checks")


def test_huggingface_connection_public_model_info():
    _require_external()

    from huggingface_hub import HfApi

    api = HfApi()
    model_info = api.model_info("google-bert/bert-base-uncased")
    assert model_info.id == "google-bert/bert-base-uncased"


def test_wandb_connection_authenticated_viewer():
    _require_external()

    if not os.getenv("WANDB_API_KEY"):
        pytest.skip("WANDB_API_KEY is required for W&B connectivity test")

    api_key = os.getenv("WANDB_API_KEY")
    payload = json.dumps(
        {
            "query": "query Viewer { viewer { id entity } }",
            "variables": {},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        status_code = response.getcode()
        body = json.loads(response.read().decode("utf-8"))

    assert status_code == 200, f"Unexpected W&B HTTP status: {status_code}"
    assert "errors" not in body, f"W&B GraphQL error: {body.get('errors')}"
    viewer = (body.get("data") or {}).get("viewer") or {}
    assert viewer.get("id"), (
        "W&B connectivity/authentication check failed.\n"
        f"response body:\n{json.dumps(body, indent=2)}\n"
    )


def test_modal_connection_cli_auth():
    _require_external()

    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        pytest.skip("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are required for Modal connectivity test")

    # `whoami` is not available on all Modal CLI versions. `app list` is a
    # stable auth-required command and fails when credentials are invalid.
    cmd = [sys.executable, "-m", "modal", "app", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Compatibility fallback for older/newer command layouts.
    if result.returncode != 0 and "No such command" in (result.stderr or ""):
        fallback_cmd = [sys.executable, "-m", "modal", "profile", "current"]
        result = subprocess.run(
            fallback_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert result.returncode == 0, (
        "Modal connectivity check failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
