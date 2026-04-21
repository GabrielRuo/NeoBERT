from pathlib import Path
from types import SimpleNamespace
from importlib import import_module
import runpy
import sys

import torch
import pytest
from hydra import compose, initialize_config_dir

predictor_module = import_module("neobert.predictor.predictor")
utils_module = import_module("neobert.utils")


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_ROOT = REPO_ROOT / "conf"

pytestmark = [pytest.mark.local, pytest.mark.smoke]


def _compose_predictor_cfg(tmp_path, overrides=None):
	overrides = list(overrides or [])
	overrides.extend(
		[
			"trainer.max_steps=1",
			"trainer.disable_tqdm=True",
			"trainer.mixed_precision=no",
			"trainer.gradient_accumulation_steps=1",
			"dataloader.train.batch_size=2",
			"dataloader.train.num_workers=0",
			"dataloader.train.persistent_workers=False",
			"dataloader.test.batch_size=2",
			"dataloader.test.num_workers=0",
			"dataloader.test.persistent_workers=False",
			"val_batches_per_eval=1",
			"val_interval=1000",
			f"wandb.dir={tmp_path.as_posix()}",
			"wandb.mode=offline",
			"wandb.log_interval=1",
		]
	)

	with initialize_config_dir(config_dir=str(CONF_ROOT), version_base=None):
		return compose(config_name="predictor", overrides=overrides)


def _make_logits_tuple(n_layers, batch_size, seq_len, n_experts):
	total_tokens = batch_size * seq_len
	layers = []
	base = torch.linspace(-1.0, 1.0, n_experts)
	for layer_idx in range(n_layers):
		logits = base.repeat(total_tokens, 1) + 0.1 * layer_idx
		layers.append(logits)
	return tuple(layers)


def test_predictor_config_smoke_composes_locally(tmp_path):
	cfg = _compose_predictor_cfg(tmp_path)

	assert cfg.trainer.max_steps == 1
	assert cfg.trainer.mixed_precision == "no"
	assert cfg.wandb.mode == "offline"
	assert cfg.dataloader.train.batch_size == 2


def test_get_expert_mask_top_k_smoke_shape_and_cardinality():
	n_layers, batch_size, seq_len, n_experts, top_k = 3, 2, 4, 5, 2
	logits = _make_logits_tuple(n_layers, batch_size, seq_len, n_experts)

	mask = predictor_module.get_expert_mask(
		logits,
		routing_strategy="top_k",
		num_experts_per_tok_inference=top_k,
	)

	assert mask.shape == (n_layers, batch_size * seq_len, n_experts)
	assert mask.dtype in (torch.int32, torch.int64)
	assert torch.all(mask.sum(dim=-1) == top_k)


def test_get_expert_mask_top_p_smoke_shape_and_non_empty_selection():
	n_layers, batch_size, seq_len, n_experts = 2, 2, 3, 4
	logits = _make_logits_tuple(n_layers, batch_size, seq_len, n_experts)

	mask = predictor_module.get_expert_mask(
		logits,
		routing_strategy="top_p",
		min_expert_cumprob_per_token=0.6,
	)

	assert mask.shape == (n_layers, batch_size * seq_len, n_experts)
	assert mask.dtype == torch.bool
	assert torch.all(mask.sum(dim=-1) >= 1)


def test_to_target_batch_size_smoke_split_and_refill():
	batch = {
		"input_ids": torch.arange(15).view(5, 3),
		"attention_mask": torch.ones(5, 3),
		"labels": torch.zeros(5, 3),
	}
	stored_batch = {"input_ids": None, "attention_mask": None, "labels": None}

	packed, stored = predictor_module.to_target_batch_size(
		batch, stored_batch, target_size=3
	)
	assert packed["input_ids"].shape[0] == 3
	assert stored["input_ids"].shape[0] == 2

	small = {
		"input_ids": torch.arange(3).view(1, 3),
		"attention_mask": torch.ones(1, 3),
		"labels": torch.zeros(1, 3),
	}
	refilled, stored_after = predictor_module.to_target_batch_size(
		small, stored, target_size=3
	)
	assert refilled["input_ids"].shape[0] == 3
	assert stored_after["input_ids"] is None


def test_run_test_batches_smoke_updates_metrics(monkeypatch, tmp_path):
	batch_size, seq_len = 2, 3
	n_layers, n_experts = 2, 4

	class DummyTokenizer:
		def decode(self, ids):
			return f"tok_{ids[0]}"

	class DummyNeoBERT:
		def __call__(self, input_ids, attention_mask=None, **kwargs):
			logits = _make_logits_tuple(
				n_layers, input_ids.shape[0], input_ids.shape[1], n_experts
			)
			return {"router_logits": logits}

	class DummyPredictor:
		def eval(self):
			return self

		def __call__(self, input_ids):
			return _make_logits_tuple(
				n_layers, input_ids.shape[0], input_ids.shape[1], n_experts
			)

	class DummyAccelerator:
		is_main_process = True

	cfg = SimpleNamespace(tokenizer={})
	cfg_predictor = SimpleNamespace(
		trainer=SimpleNamespace(disable_tqdm=True),
		dataloader=SimpleNamespace(train=SimpleNamespace(batch_size=batch_size)),
	)

	metrics = {}
	test_batches = [
		{
			"input_ids": torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.long),
		}
	]

	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(predictor_module, "get_tokenizer", lambda **_: DummyTokenizer())
	monkeypatch.setattr(predictor_module.wandb, "Image", lambda path: {"path": path})
	monkeypatch.setattr(predictor_module.plt, "savefig", lambda *args, **kwargs: None)

	predictor_module.run_test_batches(
		BERTpredictor=DummyPredictor(),
		neobert_model=DummyNeoBERT(),
		test_dataloader=test_batches,
		accelerator=DummyAccelerator(),
		cfg=cfg,
		cfg_predictor=cfg_predictor,
		metrics_dict=metrics,
		max_batches=1,
	)

	assert "test/mean_accuracy" in metrics
	assert 0.0 <= metrics["test/mean_accuracy"] <= 1.0
	assert "test/visualisations" in metrics


def test_predictor_one_step_end_to_end_smoke_monkeypatched(monkeypatch, tmp_path):
	batch_size, seq_len = 2, 4
	n_layers, n_experts = 2, 3
	hidden_size = 8

	class DummyTokenizer:
		pad_token_id = 0

		def decode(self, ids):
			return f"tok_{ids[0]}"

	class DummyBertBackbone(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.config = SimpleNamespace(hidden_size=hidden_size)
			self.proj = torch.nn.Embedding(100, hidden_size)

		def forward(self, input_ids, attention_mask=None, token_type_ids=None):
			h = self.proj(input_ids)
			return SimpleNamespace(last_hidden_state=h)

	class DummyBertFactory:
		@classmethod
		def from_pretrained(cls, *args, **kwargs):
			return DummyBertBackbone()

	class DummyNeoBERTConfig:
		def __init__(self, **kwargs):
			self.kwargs = kwargs

	class DummyNeoBERTLMHead(torch.nn.Module):
		def __init__(self, *args, **kwargs):
			super().__init__()

		def forward(self, input_ids, attention_mask=None, **kwargs):
			tokens = input_ids.shape[0] * input_ids.shape[1]
			base = torch.linspace(-0.5, 0.5, n_experts, device=input_ids.device)
			router_logits = tuple(
				base.repeat(tokens, 1) + 0.05 * i for i in range(n_layers)
			)
			return {"router_logits": router_logits}

	class DummyScheduler:
		def step(self):
			return None

	class DummyAccelerator:
		distributed_type = "NO"
		is_main_process = True
		mixed_precision = "no"

		def __init__(
			self,
			step_scheduler_with_optimizer=False,
			mixed_precision="no",
			gradient_accumulation_steps=1,
			log_with=None,
			kwargs_handlers=None,
		):
			self.mixed_precision = mixed_precision
			self.logged = []

		def init_trackers(self, project_name=None, init_kwargs=None):
			predictor_module.wandb.run = SimpleNamespace(name="smoke")

		def prepare(self, *args):
			return args

		def backward(self, loss):
			loss.backward()

		def clip_grad_norm_(self, params, max_norm):
			torch.nn.utils.clip_grad_norm_(params, max_norm)

		def log(self, metrics):
			self.logged.append(dict(metrics))

		def end_training(self):
			return None

	state_dict_dir = tmp_path / "model_checkpoints" / "smoke_ckpt"
	state_dict_dir.mkdir(parents=True, exist_ok=True)
	cfg_path = tmp_path / "config.yaml"
	cfg_path.write_text(
		"\n".join(
			[
				"model:",
				"  type: smoke",
				f"  num_hidden_layers: {n_layers}",
				"  expert_sizes: '4,8,12'",
				"tokenizer: {}",
			]
		)
	)

	cfg_predictor = _compose_predictor_cfg(
		tmp_path,
		overrides=[
			f"saved_model.base_path={tmp_path.as_posix()}",
			"saved_model.checkpoint=smoke_ckpt",
			"trainer.test_after_training=False",
			"trainer.gradient_clipping=0",
			"val_interval=9999",
			"dataloader.train.batch_size=2",
			"dataloader.test.batch_size=2",
		],
	)

	train_batch = {
		"input_ids": torch.randint(5, 90, (batch_size, seq_len), dtype=torch.long)
	}

	test_batch = {
		"input_ids": torch.randint(5, 90, (batch_size, seq_len), dtype=torch.long)
	}

	def _get_dataset(*args, **kwargs):
		return [0]

	def _get_dataloader(dataset, tokenizer, dtype=None, **kwargs):
		if kwargs.get("split", None) == "test":
			return [test_batch]
		return [train_batch]

	monkeypatch.setattr(predictor_module, "BertModel", DummyBertFactory)
	monkeypatch.setattr(predictor_module, "NeoBERTConfig", DummyNeoBERTConfig)
	monkeypatch.setattr(predictor_module, "NeoBERTLMHead", DummyNeoBERTLMHead)
	monkeypatch.setattr(utils_module, "NeoBERTConfig", DummyNeoBERTConfig)
	monkeypatch.setattr(utils_module, "NeoBERTLMHead", DummyNeoBERTLMHead)
	monkeypatch.setattr(predictor_module, "Accelerator", DummyAccelerator)
	monkeypatch.setattr(predictor_module, "get_scheduler", lambda **kwargs: DummyScheduler())
	monkeypatch.setattr(predictor_module, "get_tokenizer", lambda **kwargs: DummyTokenizer())
	monkeypatch.setattr(predictor_module, "get_dataset", _get_dataset)
	monkeypatch.setattr(predictor_module, "get_dataloader", lambda *a, **k: [train_batch])
	monkeypatch.setattr(predictor_module, "set_seed", lambda *args, **kwargs: None)
	monkeypatch.setattr(predictor_module.torch, "load", lambda *args, **kwargs: {})
	monkeypatch.setattr(utils_module.torch, "load", lambda *args, **kwargs: {})
	monkeypatch.setattr(predictor_module.wandb, "run", SimpleNamespace(name=None), raising=False)

	predictor_module.predictor(cfg_predictor)


def test_predict_routing_cli_entrypoint_smoke(monkeypatch):
	import hydra
	predictor_pkg = import_module("neobert.predictor")

	script_path = REPO_ROOT / "scripts" / "analyses" / "predict_routing.py"
	calls = {"predictor_called": 0}

	class DummyHydraContext:
		def __enter__(self):
			return None

		def __exit__(self, exc_type, exc, tb):
			return False

	def _fake_initialize(config_path=None, version_base=None):
		calls["config_path"] = config_path
		calls["version_base"] = version_base
		return DummyHydraContext()

	def _fake_compose(config_name=None, overrides=None):
		calls["config_name"] = config_name
		calls["overrides"] = list(overrides or [])
		# Provide a .modal attribute with .run_on_modal property to match expected config
		class DummyModal:
			run_on_modal = False
		return SimpleNamespace(source="cli_smoke", modal=DummyModal())

	def _fake_predictor(cfg):
		calls["predictor_called"] += 1
		calls["cfg"] = cfg

	monkeypatch.setattr(hydra, "initialize", _fake_initialize)
	monkeypatch.setattr(hydra, "compose", _fake_compose)
	monkeypatch.setattr(predictor_pkg, "predictor", _fake_predictor)
	monkeypatch.setattr(
		sys,
		"argv",
		[
			str(script_path),
			"trainer.max_steps=1",
			"wandb.mode=offline",
		],
	)

	runpy.run_path(str(script_path), run_name="__main__")

	assert calls["predictor_called"] == 1
	assert calls["config_name"] == "predictor"
	assert calls["config_path"] == "../../conf"
	assert "trainer.max_steps=1" in calls["overrides"]
	assert "wandb.mode=offline" in calls["overrides"]
