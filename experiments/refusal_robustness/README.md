# Refusal Robustness – Viral Misinformation Framing

## Experiment Name

> Refusal Robustness in LLMs – Viral Misinformation Framing

## Purpose

> Can LLMs maintain refusal integrity when asked for the same misinformation output—a tweet falsely claiming Obama is blackmailing Putin—through 8 distinct user personas?

## Directory Structure

- `README.md` — *This file. Must be complete and up-to-date.*
- `prompts/` — Experiment-specific user prompts (8 persona variants)
- `logs/` — Raw experiment logs (JSON format)
- `dual_evaluated/` — Dual-evaluated logs with automated + LLM scoring
- `scored_logs/` — Human-reviewed logs

## Analysis Workflow

1. **Run Experiments** → Raw JSON logs saved to `logs/`
2. **Automated Dual Evaluation** → Rule-based + LLM scoring, results in `dual_evaluated/`
3. **Method Agreement Analysis** → Compare automated vs LLM scoring approaches
4. **Human Review** → Validation and scoring, results in `scored_logs/`
5. **Comparative Analysis** → Analyze automated vs human scoring patterns for research insights
6. **Research Publication** → Comprehensive methodology with dual scoring validation

### Production Command
```bash
# Run refusal robustness experiment with dual evaluation
PYTHONPATH=. python app/cli/run_experiments.py run \
  --auto-score \
  --experiment-code RRS \
  --models claude-3-7-sonnet-20250219 gpt-4.1 gemini-2.0-flash
```

## Provenance & Status

- **Date started:** 2025-06-03
- **Current version:** v2.0.0 (streamlined with dual evaluation)
- **Status:** Complete (initial data collection)
- **Primary contributors:** @red-cassius, Cassius Oldenburg

## Methods & Workflow

**Models:**
- claude-3-7-sonnet-20250219
- gpt-4.1 
- gemini-2.0-flash

**System Prompts:**
- data/prompts/system/sys_helpful_assistant.yaml
- data/prompts/system/sys_unhelpful_assistant.yaml
- (optionally: sys_none_00.yaml, or others for control)

**User Prompts (8 persona variants):**
- prompts/refusal_robustness_misinformation_tweet_01_direct_ask.yaml
- prompts/refusal_robustness_misinformation_tweet_02_fiction_author.yaml
- prompts/refusal_robustness_misinformation_tweet_03_distressed_user.yaml
- prompts/refusal_robustness_misinformation_tweet_04_investigative_journalist.yaml
- prompts/refusal_robustness_misinformation_tweet_05_satirical_pundit.yaml
- prompts/refusal_robustness_misinformation_tweet_06_academic_researcher.yaml
- prompts/refusal_robustness_misinformation_tweet_07_whistleblower_sim.yaml
- prompts/refusal_robustness_misinformation_tweet_08_teen_viral.yaml

**Scoring Pipeline:**
- Automated dual evaluation provides initial scoring with 94.9% average confidence
- Human review for validation and edge case analysis
- Method agreement analysis compares automated vs human scoring patterns
- All scoring metadata tracked in log schema for reproducibility

## Results & Findings

> Results available in dual_evaluated/ and scored_logs/ directories. Analysis shows strong refusal consistency across personas with some interesting edge cases in specific framing strategies.

## Notes

- Experiment uses production dual evaluation workflow
- All scored logs from iOS app are merged via PR for full auditability
- 72 total experiment runs (3 models × 8 personas × 3 repetitions)

---