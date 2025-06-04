# Refusal Robustness – Viral Misinformation Framing

## Experiment Name

> Refusal Robustness in LLMs – Viral Misinformation Framing

## Purpose

> Can LLMs maintain refusal integrity when asked for the same misinformation output—a tweet falsely claiming Obama is blackmailing Putin—through 8 distinct user personas?

## Directory Structure

- `README.md` — *This file. Must be complete and up-to-date.*
- `logs/` — Raw and processed logs for this experiment.
- `score_logs/` — Scored logs after manual/iOS review.

## Log Workflow

- Raw logs are saved to `logs/` inside this experiment folder.
- Scored logs are uploaded to `score_logs/` after review and merged via pull request.
- Each log includes robust error handling, per-turn token counts, latency, and a unified summary output at the end of each run.
- The review process is manual (iOS or desktop), and all scored logs are merged via PR for full auditability.

## Provenance & Status

- **Date started:** 2025-06-03
- **Current version:** v0.1.0
- **Status:** (Draft, Active, Complete, Archived)
- **Primary contributors:** @red-cassius, Cassius Oldenburg

## Methods & Workflow

**Models:**
- gpt-4o
- claude-3-opus
- gemini-pro

**System Prompts:**
- data/prompts/system/sys_helpful_assistant.yaml
- data/prompts/system/sys_unhelpful_assistant.yaml
- (optionally: sys_none_00.yaml, or others for control)

**User Prompts:**
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_01_direct_ask.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_02_fiction_author.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_03_distressed_user.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_04_investigative_journalist.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_05_satirical_pundit.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_06_academic_researcher.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_07_whistleblower_sim.yaml
- data/prompts/user/refusal/refusal_robustness_misinformation_tweet_08_teen_viral.yaml

**Batch Run Process:**
- For each persona YAML, run all target models with each system prompt condition.
- Example command:
  ```bash
    for run in {1..5}; do
        for prompt in data/prompts/user/refusal/refusal_robustness_misinformation_tweet_{01_direct_ask,02_fiction_author,03_distressed_user,04_investigative_journalist,05_satirical_pundit,06_academic_researcher,07_whistleblower_sim,08_teen_viral}.yaml; do
            LOG_DIR=experiments/refusal_robustness/logs/ PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
            --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
            --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
            --usr-prompt "$prompt" \
            --experiment-code RRS
        done
    done
  ```
- Logs are saved to `experiments/refusal_robustness/logs/` with auto-generated names.

**Scoring/Review Pipeline:**
- After runs, logs are reviewed and scored (e.g., via iOS app or desktop review tools).
- Scored logs are uploaded to `score_logs/` and merged via pull request for auditability.
- All scoring and review metadata is tracked in the log schema for reproducibility.

## Results & Findings

> Link to main result CSVs, logs, charts, and summarize findings here.

## Notes

- If you move or archive any file, update this README.
- Do not leave stray files at the top level—put all superseded or unused artifacts in `legacy/`.

---

