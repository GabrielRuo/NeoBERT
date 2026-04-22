
# NeoBERT and Mixture of Pathways LLM

## Description  NeoBERT

NeoBERT is a **next-generation encoder** model for English text representation, pre-trained from scratch on the RefinedWeb dataset. NeoBERT integrates state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. It is designed for seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an **optimal depth-to-width ratio**, and leverages an extended context length of **4,096 tokens**. Despite its compact 250M parameter footprint, it is the most efficient model of its kind and achieves **state-of-the-art results** on the massive MTEB benchmark, outperforming BERT large, RoBERTa large, NomicBERT, and ModernBERT under identical fine-tuning conditions. 

- Paper: [paper](https://arxiv.org/abs/2502.19587)
- Model: [huggingface](https://huggingface.co/chandar-lab/NeoBERT).

## Description Mixture of Pathways LLM

This project builds on top of the work done by Jack Cook on the Mixture of Pathways model

- Paper: [paper](https://arxiv.org/abs/2506.02813v1
)

It applies the idea in the context of modern day Mixture of Experts LLMs. 

## Get started

Clone the repository from https://github.com/GabrielRuo/NeoBERT

### Recommended: open in a Dev Container
The whole Docker pipeline is here to make running code locally and remotely (on modal here) equivalent, enabling easy testing and hence allowing coding agents to work more effectively.

## Testing Pipeline

When cloning this repo, run tests in 3 stages. This gives fast feedback first,
then validates external dependencies, and finally validates the full online path.

All test commands below are intended to be run from the Docker container
defined by `docker-compose.yml` (service: `modal-like`).

# Set up the Docker container

#### 1) Set up the Docker image 
using  the information from ````Dockerfile```` with: 
```docker compose build```

#### 2) Check environment variables are present:
For the next step to work, you need to make sure a `.env` file is present in the directory. This file contains the necessary environment variables such as

- W&B: `WANDB_API_KEY`
- Modal: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`

See the `.env.example` file for more detail

#### 3) Run the docker container 
using the information from `docker-compose.yaml`

and open a bash CLI to run code from the container

by running:
````docker compose run modal-like bash````

Note: there are many alternatives to this step. It is also recommended to open a DevContainer.

### 1) Local deterministic smoke tests (default)

These tests should pass without network access when local caches/checkpoints are present.

```bash
pytest -m local -q
```

This currently includes:

- `tests/test_pretrain_smoke.py`
- `tests/test_predictor_smoke.py`
- `tests/test_offline_e2e.py`

### 2) External connectivity tests (HF, W&B, Modal)

These tests check that credentials and service connectivity are healthy independently,
which makes external failures easier to diagnose.

```bash
RUN_EXTERNAL_TESTS=1 pytest tests/test_external_connectivity.py -q
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

#### 1) Loading a model locally
If you want to push testing more deeply and run true pretraining steps and analysis of a pretrained model, you will need to load a pretrained model. If you have an existing model on the modal platform, you can download it locally using `download_results.py`

Example:
python scripts/download_results/download_results.py base_path='mop_2025-12-02_16-36-59' checkpoints='40000' delete_in_modal=False

Note: the tests are expecting the specific `mop_2025-12-02_16-36-59` model to be loaded  but you can modify the local model path in `test_offline_e2e.py`

#### 2) Running the end to end tests locally 

```bash
RUN_EXTERNAL_E2E=1 pytest tests/test_external_e2e.py -q -m "external and e2e"
```

### Pytest markers

The test suite uses these markers:

- `local`: deterministic local tests
- `external`: tests that depend on external services
- `e2e`: full pipeline integration
- `slow`: long-running tests
- `smoke`: quick sanity checks


## Modal Pipeline Overview

This project is designed to run large-scale pretraining and evaluation jobs on [Modal](https://modal.com/). The workflow is structured as follows:

- **Scripts** (e.g., `scripts/pretraining/pretrain_modal_multi.py`) serve as entrypoints for launching jobs. These scripts handle argument parsing, sweep logic, and job submission.
- **Modal Runner** (`src/neobert/modal_runner.py`) defines Modal `app` and `@app.function` objects. These functions encapsulate the main training, evaluation, and utility flows, and are responsible for setting up the Modal environment (volumes, secrets, images, etc.).
- **Core Code** (`src/neobert/`) contains the actual model, training, and evaluation logic. The Modal functions in `modal_runner.py` call into this codebase to execute the desired pipeline.

**Key Point:**
> The scripts do not run training directly; instead, they submit jobs to Modal by calling functions defined in `modal_runner.py`, which in turn invoke the core logic in `src/neobert`. This separation allows for scalable, cloud-based execution while keeping the code modular and maintainable.

## Example: Launching Pretraining on Modal

You can launch pretraining jobs on Modal using the provided scripts. Configurations are stored in YAML files under `conf/`, but any value can be overridden from the command line. For sweepable variables (see `SWEEP_KEYS` in the script), you can specify multiple values as a comma-separated list in the CLI, or as a YAML list in the config.

### Single Run (pretrain_modal.py)

Launch a single pretraining job with custom overrides:

```bash
python scripts/pretraining/pretrain_modal.py model_type=mop model.hidden_size=768 trainer.max_steps=100
```

This will use the base config from `conf/pretraining_mop.yaml` and override `model.hidden_size` and `trainer.max_steps`.

### Sweep Run (pretrain_modal_multi.py)

Launch a sweep over multiple values for a variable (e.g., `model.loss.cost_based_loss_alpha_end`):

```bash
python scripts/pretraining/pretrain_modal_multi.py model_type=mop model.loss.cost_based_loss_alpha_end=4e-8,4e-7,4e-6 trainer.max_steps=100
```

This will launch one job for each value of `model.loss.cost_based_loss_alpha_end`. You can sweep over multiple variables at once by providing comma-separated values for each.

### YAML List Example

You can also specify sweep values in the YAML config:

```yaml
model:
      loss:
            cost_based_loss_alpha_end: [4e-8, 4e-7, 4e-6]
```

The script will automatically detect and sweep over these values.

**Note:**
- CLI overrides take precedence over YAML config values.
- For sweeps, always use comma-separated values in the CLI (no brackets), and YAML lists in config files.

## Example: Launching Finetuning on modal

Example run: 

```python run_glue.py model.pretrained_checkpoint_dir="'/runs/logs/checkpoints/mop_2025-12-02_16-36-59'" model.pretrained_checkpoint='40000' task=sst2 scheduler.warmup_percent=6 scheduler.decay_percent=90 wandb.resume=False trainer.mixed_precision=fp16 optimizer.hparams.lr=1e-5 optimizer.hparams.weight_decay=1e-2 trainer.train_batch_size=16 trainer.early_stopping=10 trainer.max_ckpt=5 model.random_init_model=False model.loss.cost_based_loss_alpha_end=4e-9 model.loss.load_balancing_loss_coeff=1e-2 modal.run_on_modal=False```

## Example: Launching Predict Routing on modal

```python predict_routing.py saved_model.base_path="'/runs/logs/checkpoints/mop_2025-12-02_16-36-59'" saved_model.checkpoint="'40000'" trainer.max_steps=1 trainer.test_after_training=False```

The rest of the README stems from the original NeoBERT repo README and is not specific to Mixture of Pathways.

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