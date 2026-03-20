from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from datasets import Dataset
import pytest

from neobert.dataset import dataset as dataset_module


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
CACHED_MINIPILE_PATH = "/data/.pathways_cache/jeankaddourminipiletrain_100"

pytestmark = [pytest.mark.local, pytest.mark.smoke]


def _compose_pretrain_cfg(tmp_path, overrides=None, dataset_path=None):
    overrides = list(overrides or [])
    _ds_path = (
        dataset_path.as_posix() if hasattr(dataset_path, "as_posix") else dataset_path
    ) or CACHED_MINIPILE_PATH
    overrides.extend(
        [
            "trainer.max_steps=1",
            "trainer.resume=False",
            "trainer.save_model=False",
            "trainer.disable_tqdm=True",
            "trainer.mixed_precision=no",
            "trainer.gradient_accumulation_steps=1",
            "dataloader.train.batch_size=1",
            "dataloader.train.num_workers=0",
            "dataloader.train.persistent_workers=False",
            f"trainer.dir={tmp_path.as_posix()}",
            f"dataset.train.path_to_disk={_ds_path}",
            "wandb.mode=offline",
            "wandb.log_interval=1",
        ]
    )

    with initialize_config_dir(
        config_dir=str(CONF_ROOT),
        version_base=None,
    ):
        return compose(config_name="pretraining_mop", overrides=overrides)


def test_pretrain_config_smoke_composes_locally(tmp_path):
    cfg = _compose_pretrain_cfg(tmp_path)

    assert cfg.trainer.max_steps == 1
    assert cfg.trainer.mixed_precision == "no"
    assert cfg.wandb.mode == "offline"
    assert cfg.dataset.train.path_to_disk == CACHED_MINIPILE_PATH


def test_get_dataset_prefers_configured_path_to_disk(monkeypatch, tmp_path):
    cached_ds_path = tmp_path / "cached_ds"
    Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}).save_to_disk(
        str(cached_ds_path)
    )

    cfg = SimpleNamespace(
        tokenizer={},
        dataset=SimpleNamespace(column="text"),
    )

    called = {"load_dataset": False}

    def _fail_load_dataset(*args, **kwargs):
        called["load_dataset"] = True
        raise AssertionError("load_dataset should not be called when path_to_disk is set")

    monkeypatch.setattr(dataset_module, "load_dataset", _fail_load_dataset)

    dataset = dataset_module.get_dataset(
        cfg,
        hf_path="JeanKaddour/minipile",
        split="train",
        num_samples=100,
        path_to_disk=str(cached_ds_path),
    )

    assert len(dataset) == 1
    assert called["load_dataset"] is False


def test_pretrain_trainer_end_to_end_smoke(monkeypatch, tmp_path):
    """Build a 4-sample synthetic dataset and run exactly one MoP training step
    without touching the network (the google-bert/bert-base-uncased tokenizer must already be
    cached in HF_HOME, which is /cache/hf inside the container)."""
    import torch
    from neobert.pretraining.trainer import trainer

    # 1. Synthetic tokenised dataset: 4 sequences × 64 token-ids (safe BERT vocab range)
    rng = torch.Generator()
    rng.manual_seed(42)
    input_ids = torch.randint(1000, 25000, (4, 64), generator=rng).tolist()
    ds = Dataset.from_dict({"input_ids": input_ids})
    ds_path = tmp_path / "synth_dataset"
    ds.save_to_disk(str(ds_path))

    # 2. Replace torch.compile with a no-op to avoid compilation overhead on CPU
    monkeypatch.setattr(torch, "compile", lambda m, **kw: m)

    # 3. Enforce offline mode – the tokenizer must be pre-cached in HF_HOME
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    # 4. Compose config with a tiny model to keep the forward pass fast
    cfg = _compose_pretrain_cfg(
        tmp_path,
        dataset_path=ds_path,
        overrides=[
            # Tiny architecture – hidden_size must be divisible by num_attention_heads
            "model.hidden_size=8",
            "model.num_hidden_layers=2",
            "model.num_attention_heads=2",
            # 3 experts: 0=identity, 24 and 48 produce valid SwiGLU layers for hidden_size=8
            "model.expert_sizes='0,24,48'",
            "model.num_experts_per_tok_training=2",
            "model.num_experts_per_tok_inference=1",
            "dataloader.train.pin_memory=False",
        ],
    )

    # 5. Run exactly one training step – should complete without raising
    trainer(cfg)
