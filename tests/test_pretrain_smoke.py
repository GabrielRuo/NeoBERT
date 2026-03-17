from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from datasets import Dataset

from neobert.dataset import dataset as dataset_module


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"
CACHED_MINIPILE_PATH = "/data/.pathways_cache/jeankaddourminipiletrain_100"


def _compose_pretrain_cfg(tmp_path, overrides=None):
    overrides = list(overrides or [])
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
            f"dataset.train.path_to_disk={CACHED_MINIPILE_PATH}",
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
