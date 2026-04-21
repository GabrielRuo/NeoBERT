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
def test_predict_routing_full_pipeline_offline_local_assets():        # Always create the provided config.yaml file in the expected location
    default_base_path = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_BASE_PATH",
        "/tests/run_tests/mop_2025-12-02_16-36-59/",
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

    config_path = Path(default_base_path) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_content = '''
seed: 0
wandb:
    resume: never
    name: null
    project: MoP
    entity: gabriel-ruault-centralesup-lec
    tags: []
    dir: logs/wandb
    mode: online
    log_interval: 10
test:
    max_steps: 10
    buffer_size_seq: 100
    buffer_size_token: 100
dataset:
    name: minipile
    column: text
    path_to_disk: tokenized_datasets/mini_pile
    train:
        hf_path: JeanKaddour/minipile
        split: train
        num_samples: null
        streaming: false
    test:
        hf_path: JeanKaddour/minipile
        split: test
        num_samples: null
        streaming: false
tokenizer:
    pretrained_model_name_or_path: google-bert/bert-base-uncased
    trust_remote_code: true
    max_length: 128
    vocab_size: 30522
model:
    type: mop
    rope: true
    rms_norm: true
    hidden_act: swiglu
    dropout_prob: 0
    norm_eps: 1.0e-05
    embedding_init_range: 0.02
    decoder_init_range: 0.02
    classifier_init_range: 0.02
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    router_jitter_noise: 1.0e-05
    routing_strategy: top_k
    num_experts_per_tok_training: 8
    num_experts_per_tok_inference: 2
    apply_expert_dropout: false
    dropout_max_prob: null
    dropout_router_weight_threshold: null
    expert_cost_exponent: 2
    expert_sizes: 0,504, 616, 728, 840, 952, 1064, 1176
    loss:
        cost_based_loss_alpha_start: 1.0e-10
        cost_based_loss_alpha_end: 4.0e-08
        cost_based_loss_schedule_tokens: 1
        cost_based_loss_epsilon: 1.0e-05
        denominator_exponent: 1
        load_balancing_loss_coeff: 0.1
        disable_task_performance_scaling: false
        alpha_scaling: 0.0
optimizer:
    name: AdamW
    hparams:
        lr: 0.0001
        betas:
        - 0.9
        - 0.95
        eps: 1.0e-08
        weight_decay: 0.01
scheduler:
    warmup_steps: 2000
    decay_steps: 900000
    decay: cosine
trainer:
    tf32: true
    mixed_precision: bf16
    resume: true
    disable_tqdm: false
    max_steps: 40001
    gradient_accumulation_steps: 1
    gradient_clipping: 1
    accelerate:
        save_steps: 1000
        max_ckpt: 5
    model:
        save_steps: 10000
        max_ckpt: 1000
    dir: /runs/logs/checkpoints
    save_model: true
dataloader:
    train:
        num_workers: 4
        batch_size: 64
        shuffle: true
        pin_memory: true
        persistent_workers: false
    test:
        num_workers: 4
        batch_size: 32
        shuffle: false
        pin_memory: true
        persistent_workers: false
datacollator:
    mlm_probability: 0.15
    pad_to_multiple_of: 8
    '''
    with open(config_path, 'w') as f:
        f.write(config_content)

    """
    Run predict_routing with local-only model + datasets and no network access.

    Skips gracefully if required local assets (model, datasets, tokenizer) are missing.
    This test executes the real script entrypoint and is intentionally opt-in
    because it depends on pre-existing local artifacts.
    """
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]

    if importlib.util.find_spec("dotenv") is None:
        pytest.skip(
            "python-dotenv is required by scripts/analyses/predict_routing.py. "
            "Install it to run this offline E2E test."
        )

    # Check for cached tokenizer early; skip if missing to avoid subprocess failure
    tokenizer_name = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_TOKENIZER",
        "google-bert/bert-base-uncased",
    )
    try:
        AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception as exc:
        pytest.skip(
            "Offline predict_routing E2E requires a locally cached tokenizer. "
            f"Missing tokenizer {tokenizer_name!r}: {exc}"
        )

    script_path = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_SCRIPT",
        "scripts/analyses/predict_routing.py",
    )

    timeout_s = int(os.getenv("PREDICT_ROUTING_OFFLINE_E2E_TIMEOUT", "3600"))
    extra_overrides = shlex.split(os.getenv("PREDICT_ROUTING_OFFLINE_E2E_OVERRIDES", ""))

    default_base_path = os.getenv(
        "PREDICT_ROUTING_OFFLINE_E2E_BASE_PATH",
        "/tests/run_tests/mop_2025-12-02_16-36-59/",
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


    # Always create and use a random model file matching the config, using the real NeoBERT model
    import torch
    import yaml
    model_path = Path(default_base_path) / "model_checkpoints" / str(default_checkpoint) / "state_dict.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating a random NeoBERT model checkpoint at {model_path}.")

    # Import NeoBERTConfig and NeoBERTLMHead
    sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))
    from neobert.model.model import NeoBERTConfig, NeoBERTLMHead

    # Read config.yaml and extract the model section
    with open(config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    model_cfg = config_yaml["model"]
    # Normalize keys to match NeoBERTConfig signature (handle case sensitivity)
    model_cfg = {k.lower(): v for k, v in model_cfg.items()}
    # Some config keys may need renaming to match NeoBERTConfig
    if "hidden_act" in model_cfg:
        model_cfg["hidden_act"] = model_cfg["hidden_act"].capitalize()
    # Remove keys not in NeoBERTConfig
    allowed_keys = NeoBERTConfig().to_dict().keys()
    model_cfg = {k: v for k, v in model_cfg.items() if k in allowed_keys}

    # Fill in required defaults if missing
    if "vocab_size" not in model_cfg:
        model_cfg["vocab_size"] = 30522
    if "pad_token_id" not in model_cfg:
        model_cfg["pad_token_id"] = 0
    if "max_length" not in model_cfg:
        model_cfg["max_length"] = 128

    config = NeoBERTConfig(**model_cfg)
    model = NeoBERTLMHead(config)
    torch.save(model.state_dict(), model_path)

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
    """Run pretrain.py end-to-end with a local synthetic dataset in offline mode.
    
    Skips gracefully if no tokenizer is cached locally; users can pre-cache
    a tokenizer via: python -c "from transformers import AutoTokenizer; 
    AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')" in online mode,
    or run with RUN_OFFLINE_E2E=0 to skip offline tests entirely.
    """

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
