# Agent Workflow Contract

This file defines how to run cleanup workflows safely with explicit human control.

## Modes

- audit mode
  - Read-only analysis and proposals.
  - No file edits.

- apply mode
  - Applies only approved changes.
  - No implicit deletions or renames.

- commit mode
  - Commits only approved and already-applied changes.
  - No push unless explicitly requested.

## Control Commands

Use these exact commands in chat:

- START FOLDER <path>
- PROPOSE ONLY
- KEEP <path>
- DELETE <path>
- RENAME <old path> -> <new path>
- DEFER <path>
- APPLY APPROVED
- COMMIT APPROVED
- NEXT FOLDER
- SKIP FOLDER
- STOP

## Required Proposal Format

For each file, provide:

- path
- classification: Keep, Delete-candidate, Rename-candidate, Defer
- evidence: where used, or no-reference check
- risk: low, medium, high
- suggested action

## Commit Standard

- one logical change per commit
- subject format: type(scope): short summary
- body must include:
  - files changed
  - validation commands and outcomes
  - known risks

## Safety Rules

- No destructive changes without explicit approval.
- Max 10 destructive file operations per batch.
- Run relevant validation after each apply batch.
- Never revert unrelated local changes.

## Branch Safety Policy

- Never commit directly to `main` or `master`.
- All code changes must be committed on `clean_up_code`.
- If currently on `main` or `master`, switch or create branch before editing:
  - `git switch clean_up_code || git switch -c clean_up_code`
- Push only `clean_up_code`:
  - `git push -u origin clean_up_code` (first push)
  - `git push origin clean_up_code` (subsequent pushes)
- If branch switch/create fails, stop and ask the user. Do not commit.

## Pre-Commit Check

- Mandatory before every commit:
  - Run `git branch --show-current`.
  - If output is not `clean_up_code`, do not commit; switch branches first.