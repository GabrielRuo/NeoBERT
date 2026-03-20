## NeoBERT — quick guidance for AI coding agents

This project implements NeoBERT (pretraining + evaluation + utilities). The notes below are intentionally terse and actionable: when changing code, prefer small, local edits and run the exact workflow shown in the script examples.

- Big picture
  - Code lives under `src/neobert/` (core modules: `pretraining`, `glue`, `predictor`, `analysis`, `contrastive`).
  - Configuration is Hydra-based and placed in `conf/`. Many behaviors (datasets, model variants, dataloading, optimizer, scheduler) are driven by Hydra config files named like `pretraining_*.yaml`, `glue*.yaml`, etc.
  - Entry points: `scripts/pretraining/pretrain.py` (Hydra `pretraining` pipeline), other scripts under `scripts/` mirror main flows. For cloud/remote runs look at `src/neobert/modal_runner.py`.

- Important developer workflows (examples)
  - Quick local run (single-process, with Hydra overrides):
    - `python scripts/pretraining/pretrain.py dataset=refinedweb tokenizer=google model=[neobert,250M-opt] dataloader.train.batch_size=8 trainer.max_steps=100` 
      (Use small `max_steps` for fast smoke tests.)
  - Multi-GPU / cluster launch using Accelerate (example from `jobs/neobert-4096.sh`):
    - `accelerate launch --config_file <path>/conf/accelerate_deepspeed_zero2.yaml --num_processes <n> scripts/pretraining/pretrain.py <overrides...>`
  - Modal remote runs: `src/neobert/modal_runner.py` registers functions (e.g., `run_pretrain`, `run_predictor`) and expects secrets `HF_TOKEN` and `WANDB_API_KEY`. Volumes used: `neobert-runs`, `neobert-training-data`.

- Where to look for implementation patterns
  - Trainer contract: functions named `trainer(cfg: DictConfig)` are the canonical entrypoints (see `src/neobert/pretraining/trainer.py`, `glue/train.py`, `contrastive/trainer.py`). They expect an OmegaConf `DictConfig` from Hydra.
  - Predictor: `src/neobert/predictor/predictor.py` exposes `predictor(cfg)`.
  - Scripts: `scripts/pretraining/pretrain.py` wraps the `trainer` behind a Hydra main decorator; prefer editing logic inside `src/neobert/*` rather than changing scripts unless adding a new pipeline.

- Project-specific conventions and gotchas
  - Hydra is the single source of truth for runtime options. Most runtime parameters are set via config keys (e.g., `dataloader.train.batch_size`, `trainer.max_steps`, `tokenizer.max_length`). Use overrides rather than hardcoding values.
  - Models are often passed as lists in the CLI (example in `jobs/neobert-4096.sh`: `model=[neobert,250M-opt]`). Follow existing list/tuple conventions when constructing `model` overrides.
  - Accelerate + Deepspeed configs live under `conf/` (files like `accelerate_deepspeed_zero2.yaml`). Use those files as canonical examples for distributed launches.
  - Outputs/hydra.run.dir: CI/cluster jobs explicitly set `hydra.run.dir` or `trainer.dir` in job scripts — copying these patterns avoids Hydra-created directories in unexpected locations.

- External integrations to be mindful of
  - Hugging Face: HF token read from `HF_TOKEN` env var; code uses `transformers` with `trust_remote_code=True` in places.
  - Weights & Biases: `WANDB_API_KEY` required for full run; some job scripts set `wandb.mode=offline` and `wandb.dir`.
  - Modal: optional cloud runner (`modal` extra). `modal_runner.py` shows how secrets and volumes are passed.

- Quick file map (start here)
  - `conf/` — all Hydra configs and accelerate configs
  - `scripts/pretraining/pretrain.py` — standard entrypoint used by `accelerate launch` and job scripts
  - `src/neobert/pretraining/trainer.py` — main training logic; follow its `DictConfig` usage
  - `src/neobert/modal_runner.py` — Modal examples: volumes, secrets, function signatures
  - `jobs/*.sh` — real-world `accelerate launch` usage and example overrides (useful for reproducing cluster runs locally)

- Practical pointers for making edits
  - If you change a Hydra option, update the config in `conf/` and add a short comment there. Search for where that key is consumed in `src/neobert/` before changing semantics.
  - For new CLI options, prefer a Hydra config rather than adding new positional args to scripts.
  - Tests: `pyproject.toml` lists `pytest` under dev extras; create focused unit tests that import the trainer functions and run a tiny in-memory dataset.

If anything here is unclear or you want examples for a specific flow (e.g., add a new pretraining config, or run a tiny local multi-process test), tell me which flow and I will expand the instructions.

## Agent Operating Contract (Authoritative)

This section is the source of truth for autonomous coding behavior in this repository.

- Default mode: audit-first
  - Agents must start in read-only audit mode for cleanup work.
  - No file deletion, rename, or move is allowed before explicit user approval.

- Cleanup decision rubric
  - Keep: file is referenced by imports, entrypoints, tests, configs, CI, or docs.
  - Delete-candidate: no references, duplicate/copy artifact, obsolete generated file, or empty/stub file.
  - Rename-candidate: useful file with non-descriptive or inconsistent naming.
  - Defer: uncertain external usage or medium/high removal risk.

- Evidence requirement
  - Every delete or rename proposal must include concrete evidence:
    - at least one reference search result, or
    - explicit confirmation that no references were found.

- Batch size and safety
  - Apply destructive changes in small batches (max 10 files).
  - After each batch, run relevant tests or collection checks.
  - Never revert unrelated local modifications.

- Commit discipline
  - One logical change per commit.
  - Commit message format: type(scope): short summary
  - Commit body must include: files changed, verification run, and risk notes.

## Approval Protocol

The user controls execution with the following commands.

- PROPOSE ONLY
  - Agent produces recommendations with evidence; no edits.

- APPLY APPROVED
  - Agent applies only items explicitly approved by the user.

- COMMIT APPROVED
  - Agent creates a commit only for already approved changes.

- NEXT FOLDER
  - Agent moves audit to the next folder.

- SKIP FOLDER
  - Agent skips current folder without edits.

- STOP
  - Agent halts immediately and performs no further actions.

## Branch Safety Policy (Required)

- Never commit directly to `main` or `master`.
- All code changes must be committed on `clean_up_code`.
- If currently on `main` or `master`, switch or create branch before editing:
  - `git switch clean_up_code || git switch -c clean_up_code`
- Push only `clean_up_code`:
  - `git push -u origin clean_up_code` (first push)
  - `git push origin clean_up_code` (subsequent pushes)
- If branch switch/create fails, stop and ask the user. Do not commit.

## Pre-Commit Check (Mandatory)

- Run `git branch --show-current` before every commit.
- If output is not `clean_up_code`, do not commit; switch branches first.
