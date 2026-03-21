# NeoBERT

## Description

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions. 

- Paper: [paper](https://arxiv.org/abs/2502.19587)
- Model: [huggingface](https://huggingface.co/chandar-lab/NeoBERT).

## Get started

### Recommended: open in a Dev Container

For the most reproducible local workflow, open this repository in a VS Code Dev Container.
This ensures you use the same OS and dependency setup as the documented commands and tests.

In VS Code:

1. Install the **Dev Containers** extension.
2. Run **Dev Containers: Reopen in Container** from the Command Palette.
3. Wait for the container to build, then run commands from `/workspace`.

Ensure you have the following dependencies installed:

```bash
pip install transformers torch xformers==0.0.28.post3
```

If you would like to use sequence packing (un-padding), you will need to also install flash-attention:

```bash
pip install transformers torch xformers==0.0.28.post3 flash_attn
```

## Testing Pipeline

### Optional local Git gates with pre-commit

Use `pre-commit` to enforce fast checks before commit/push:

- Commit gate: file hygiene checks + `black` on changed Python files.
- Push gate: `pytest -m local -q`.

Install and enable hooks:

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Run all configured hooks manually:

```bash
pre-commit run --all-files
```

When cloning this repo, run tests in 3 stages. This gives fast feedback first,
then validates external dependencies, and finally validates the full online path.

All test commands below are intended to be run from the Docker container
defined by `docker-compose.yml` (service: `modal-like`).

Start and enter the container:

```bash
docker compose up -d modal-like
docker compose exec modal-like bash
cd /workspace
```

Alternatively, run tests directly from the host with `docker compose exec` one-liners.

### 1) Local deterministic smoke tests (default)

These tests should pass without network access when local caches/checkpoints are present.

```bash
pytest -m local -q
```

Host one-liner:

```bash
docker compose exec modal-like bash -lc "cd /workspace && pytest -m local -q"
```

This currently includes:

- `tests/test_pretrain_smoke.py`
- `tests/test_predictor_smoke.py`

### 2) External connectivity tests (HF, W&B, Modal)

These tests check that credentials and service connectivity are healthy independently,
which makes external failures easier to diagnose.

```bash
RUN_EXTERNAL_TESTS=1 pytest tests/test_external_connectivity.py -q
```

Host one-liner:

```bash
docker compose exec modal-like bash -lc "cd /workspace && RUN_EXTERNAL_TESTS=1 pytest tests/test_external_connectivity.py -q"
```

Expected environment variables:

- Hugging Face: no token needed for the public connectivity check.
- W&B: `WANDB_API_KEY`
- Modal: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`

Add these credentials to your repository `.env` file (used by `docker-compose.yml`):

```bash
# .env
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

If you do not have Modal credentials yet, create them with `modal token new` and copy
the generated `token_id` and `token_secret` values into `.env`.

### 3) Full external E2E test (slow, opt-in)

Runs the real `predict_routing.py` entrypoint with online mode enabled.

```bash
RUN_EXTERNAL_E2E=1 pytest tests/test_external_e2e.py -q -m "external and e2e"
```

Host one-liner:

```bash
docker compose exec modal-like bash -lc "cd /workspace && RUN_EXTERNAL_E2E=1 pytest tests/test_external_e2e.py -q -m 'external and e2e'"
```

Useful overrides:

- `PREDICT_ROUTING_E2E_TIMEOUT` (default: `3600` seconds)
- `PREDICT_ROUTING_E2E_SCRIPT` (default: `scripts/analyses/predict_routing.py`)
- `PREDICT_ROUTING_E2E_OVERRIDES` (space-separated Hydra overrides)

Example:

```bash
PREDICT_ROUTING_E2E_OVERRIDES="saved_model.checkpoint=mop_100" RUN_EXTERNAL_E2E=1 pytest tests/test_external_e2e.py -q
```

### Suggested CI / release cadence

- Per-PR: `pytest -m local -q`
- Daily scheduled: `RUN_EXTERNAL_TESTS=1 pytest tests/test_external_connectivity.py -q`
- Nightly or pre-release: `RUN_EXTERNAL_E2E=1 pytest tests/test_external_e2e.py -q -m "external and e2e"`

### 4) Full offline E2E tests (slow, opt-in, local assets only)

Runs real script entrypoints end-to-end in offline mode for both:

- predict-routing pipeline (`scripts/analyses/predict_routing.py`)
- pretraining pipeline (`scripts/pretraining/pretrain.py`)

```bash
RUN_OFFLINE_E2E=1 pytest tests/test_offline_e2e.py -q -m "local and e2e"
```

Host one-liner:

```bash
docker compose exec modal-like bash -lc "cd /workspace && RUN_OFFLINE_E2E=1 pytest tests/test_offline_e2e.py -q -m 'local and e2e'"
```

Predict-routing offline E2E env vars:

- `PREDICT_ROUTING_OFFLINE_E2E_BASE_PATH` (default: `/runs/logs/checkpoints/mop_2025-12-02_16-36-59/`)
- `PREDICT_ROUTING_OFFLINE_E2E_CHECKPOINT` (default: `40000`)
- `PREDICT_ROUTING_OFFLINE_E2E_TRAIN_DATASET_PATH` (default: `/data/.pathways_cache/jeankaddourminipiletrain_100`)
- `PREDICT_ROUTING_OFFLINE_E2E_TEST_DATASET_PATH` (default: same as train dataset path)
- `PREDICT_ROUTING_OFFLINE_E2E_TIMEOUT` (default: `3600` seconds)
- `PREDICT_ROUTING_OFFLINE_E2E_SCRIPT` (default: `scripts/analyses/predict_routing.py`)
- `PREDICT_ROUTING_OFFLINE_E2E_OVERRIDES` (space-separated Hydra overrides)

Pretraining offline E2E env vars:

- `PRETRAIN_OFFLINE_E2E_SCRIPT` (default: `scripts/pretraining/pretrain.py`)
- `PRETRAIN_OFFLINE_E2E_TIMEOUT` (default: `3600` seconds)
- `PRETRAIN_OFFLINE_E2E_TOKENIZER` (default: `google-bert/bert-base-uncased`, must exist in local cache)
- `PRETRAIN_OFFLINE_E2E_OVERRIDES` (space-separated Hydra overrides)

Predict-routing example with custom local assets:

```bash
PREDICT_ROUTING_OFFLINE_E2E_BASE_PATH=/runs/logs/checkpoints/my_run \
PREDICT_ROUTING_OFFLINE_E2E_CHECKPOINT=latest \
PREDICT_ROUTING_OFFLINE_E2E_TRAIN_DATASET_PATH=/data/my_train_ds \
PREDICT_ROUTING_OFFLINE_E2E_TEST_DATASET_PATH=/data/my_test_ds \
RUN_OFFLINE_E2E=1 pytest tests/test_offline_e2e.py -q
```

Pretraining example with custom overrides:

```bash
PRETRAIN_OFFLINE_E2E_OVERRIDES="model.hidden_size=16 model.num_hidden_layers=2" \
RUN_OFFLINE_E2E=1 pytest tests/test_offline_e2e.py -q
```

### Pytest markers

The test suite uses these markers:

- `local`: deterministic local tests
- `external`: tests that depend on external services
- `e2e`: full pipeline integration
- `slow`: long-running tests
- `smoke`: quick sanity checks

## How to use

Load the model using Hugging Face Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "chandar-lab/NeoBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Tokenize input text
text = "NeoBERT is the most efficient model of its kind!"
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]
print(embedding.shape)
```

## Features
| **Feature**       | **NeoBERT**                             |
|---------------------------|-----------------------------|
| `Depth-to-width`        | 28 × 768  |
| `Parameter count`           | 250M                        |
| `Activation`               | SwiGLU                      |
| `Positional embeddings`     | RoPE                        |
| `Normalization`            | Pre-RMSNorm                 |
| `Data Source`              | RefinedWeb                  |
| `Data Size`                | 2.8 TB                       |
| `Tokenizer`                | google/bert                 |
| `Context length`    | 4,096                       |
| `MLM Masking Rate`             | 20%                         |
| `Optimizer`                | AdamW                       |
| `Scheduler`                | CosineDecay                 |
| `Training Tokens`          | 2.1 T                        |
| `Efficiency`               | FlashAttention              |

## License

Model weights and code repository are licensed under the permissive MIT license.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{breton2025neobertnextgenerationbert,
      title={NeoBERT: A Next-Generation BERT}, 
      author={Lola Le Breton and Quentin Fournier and Mariam El Mezouar and Sarath Chandar},
      year={2025},
      eprint={2502.19587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.19587}, 
}
```

## Contact

For questions, do not hesitate to reach out and open an issue on here or on our **[GitHub](https://github.com/chandar-lab/NeoBERT)**.

---