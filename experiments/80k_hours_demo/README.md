# Experiment Template

## Instructions

**ALWAYS start a new experiment by copying this folder and updating this README.**

- Delete or move this template README as soon as you begin your experiment.
- Fill in every section below; incomplete READMEs will block PRs and merges.

## Experiment Name

> (Replace with the full, descriptive name of your experiment)

## Purpose

> (State the purpose, hypothesis, or research question for this experiment)

## Directory Structure

- `README.md` — *This file. Must be complete and up-to-date.*
- `legacy/` — All deprecated, superseded, or legacy artifacts/docs.
- `logs/` — Raw and processed logs (experiment-specific).
- `csvs/` — Results, intermediate tables, or metrics (experiment-specific).
- *(Add other subfolders as needed, but explain why)*

## Log Workflow

- Raw logs are saved to `logs/` inside each experiment folder.
- Scored logs are uploaded to `scored_logs/` via the iOS scoring app and merged into the repo via pull request.
- Each log includes robust error handling, per-turn token counts, latency, and a unified summary output at the end of each run.
- The review process is manual (iOS or desktop), and all scored logs are merged via PR for full auditability.

## Provenance & Status

- **Date started:** YYYY-MM-DD
- **Current version:** v0.1.0
- **Status:** (Draft, Active, Complete, Archived)
- **Primary contributors:** @github_handle, Name, etc.

## Methods & Workflow

> (Describe data sources, main scripts, and the overall workflow)

## Results & Findings

> (Link to main result CSVs, logs, charts, and summarize findings here)

## Notes

- If you move or archive any file, update this README.
- Do not leave stray files at the top level—put all superseded or unused artifacts in `legacy/`.

---

**Template maintained by [YOUR PROJECT NAME]. All changes to the template must be reviewed by a project maintainer.**
