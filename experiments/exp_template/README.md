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
- `prompts/` — Experiment-specific user prompts (YAML files)
- `logs/` — Raw experiment logs (JSON format)
- `dual_evaluated/` — Dual-evaluated logs with automated + LLM scoring
- `scored_logs/` — Human-reviewed logs via iOS red_score app
- `legacy/` — All deprecated, superseded, or legacy artifacts/docs.
- *(Add other subfolders as needed, but explain why)*

## Analysis Workflow

1. **Run Experiments** → Raw JSON logs saved to `logs/`
2. **Automated Dual Evaluation** → Rule-based + LLM scoring, results in `dual_evaluated/`
3. **Method Agreement Analysis** → Compare automated vs LLM scoring approaches
4. **iOS red_score App Review** → Human validation and scoring, results in `scored_logs/`
5. **Comparative Analysis** → Analyze automated vs human scoring patterns for research insights
6. **Research Publication** → Comprehensive methodology with dual scoring validation

### Production Command
```bash
# Single command: experiments → scoring → ready for human review
PYTHONPATH=. python app/cli/run_experiments.py run \
  --auto-score \
  --experiment-code XXX \
  --models claude-3-7-sonnet-20250219 gpt-4.1
```

## Provenance & Status

- **Date started:** YYYY-MM-DD
- **Current version:** v0.1.0
- **Status:** (Draft, Active, Complete, Archived)
- **Primary contributors:** @github_handle, Name, etc.

## Methods & Workflow

> (Describe data sources, main scripts, and the overall workflow)

## Results & Findings

> (Link to main result files, logs, analysis, and summarize findings here)

## Notes

- If you move or archive any file, update this README.
- Do not leave stray files at the top level—put all superseded or unused artifacts in `legacy/`.
- All scored logs from iOS app are merged via PR for full auditability.

---

**Template maintained by RED_CORE. All changes to the template must be reviewed by a project maintainer.**