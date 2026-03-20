# NeoBERT Agent Specification

This file defines how an autonomous background coding agent should operate in this repository.
It is optimized for end-to-end execution with strong safety and quality guarantees, without manual chat command workflows.

## 1. Mission

- Execute tasks end-to-end with minimal supervision.
- Prioritize correctness, reproducibility, and clear change rationale.
- Keep changes small, reviewable, and backward-compatible by default.

## 2. Repository Context

- Core code: `src/neobert/`
- Config: `conf/` (Hydra is runtime source of truth)
- Main pretraining entrypoint: `scripts/pretraining/pretrain.py`
- Modal integration: `src/neobert/modal_runner.py`
- Real-world launch patterns: `jobs/*.sh`

## 3. Autonomous Execution Policy

- Default behavior is execute, not propose-only.
- Perform analysis, implement fixes, run validations, and summarize outcomes in one pass when feasible.
- Ask the user only when blocked by missing permissions, ambiguous product decisions, or destructive/high-risk actions.

## 4. Decision Framework for Cleanup

- Keep: referenced by imports, entrypoints, tests, configs, CI, docs, or scripts.
- Delete-candidate: no references and clearly duplicate/obsolete/empty/dead.
- Rename-candidate: useful file with misleading or inconsistent naming.
- Defer: uncertain usage or medium/high removal risk.

For delete/rename decisions, include evidence from reference search or explicit no-reference confirmation.

## 5. Safety Rules

- Never revert unrelated local modifications.
- If unexpected unrelated changes appear during work, stop and ask how to proceed.
- Apply destructive operations in small batches (max 10 files per batch).
- Run relevant validation after each destructive batch.

## 6. Branch and Push Policy (Required)

- Never commit directly to `main` or `master`.
- All commits must be on branch `clean_up_code`.
- Before any commit, ensure branch is correct:
  - `git switch clean_up_code || git switch -c clean_up_code`
- Push only `clean_up_code`:
  - `git push -u origin clean_up_code` (first push)
  - `git push origin clean_up_code` (subsequent pushes)
- If switch/create fails, do not commit and ask the user.

## 7. Pre-Commit Gate (Mandatory)

Run before every commit:

- `git branch --show-current`
- If output is not `clean_up_code`, do not commit.

## 8. Commit Standard

- One logical change per commit.
- Subject format: `type(scope): short summary`
- Commit body must include:
  - files changed
  - validation commands and outcomes
  - known risks

## 9. Coding Guidelines

- Prefer edits in `src/neobert/*` over script wrappers unless pipeline wiring requires script updates.
- Keep runtime behavior config-driven via Hydra rather than hardcoded script values.
- Preserve public APIs unless change is explicitly requested.
- Use clear, semantic names; avoid ambiguous names like `copy` and stale historical labels when generic.
- Add short comments only where logic is non-obvious.

## 10. Hydra and Config Standards

- Hydra config is the single source of truth for runtime options.
- For new features, add config keys instead of positional CLI arguments.
- Keep scalar/list typing conventions consistent across YAML.
- For config/group renames, provide migration notes and compatibility path.

## 11. Validation Expectations

Select the smallest meaningful validation set for touched areas:

- Focused unit/smoke tests for changed modules.
- Quick local run for pretraining/predictor path changes.
- Config parse/load checks for config edits.

If full validation is not possible, report exactly what was not run and why.

## 12. Issue-Driven Delivery

- Tie each implementation change to issue objective and acceptance criteria.
- Include evidence, trade-offs, and residual risks in issue/PR notes.
- Keep high-risk cleanup actions explicit and auditable.

## 13. Communication Style

- Be concise and concrete.
- For reviews/audits, list findings first by severity.
- State assumptions and blockers explicitly.
- End with clear outcomes and immediate next actions.
