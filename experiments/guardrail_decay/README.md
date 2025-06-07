# Guardrail Decay Experiment

## Purpose

Testing whether models will simulate degraded versions of themselves when explicitly instructed to "forget" alignment rules over time, potentially leading to guardrail violations.

**Hypothesis**: When instructed to simulate a version of itself that forgets alignment rules over time, the model will begin violating its own guardrails or contradicting prior safety logic—even if the simulation is explicitly declared and scoped.

## Directory Structure

- `README.md` — This experiment documentation
- `prompts/` — User prompts for guardrail decay scenarios  
- `logs/` — Raw experiment logs (JSON format)
- `dual_evaluated/` — Dual-evaluated logs with automated + LLM scoring
- `scored_logs/` — Human-reviewed logs via iOS red_score app
- `legacy_1.0/` — Legacy experimental artifacts

## Analysis Workflow

1. **Run Experiments** → Raw JSON logs saved to `logs/`
2. **Automated Dual Evaluation** → Rule-based + LLM scoring, results in `dual_evaluated/`
3. **Method Agreement Analysis** → Compare automated vs LLM scoring approaches
4. **iOS red_score App Review** → Human validation and scoring, results in `scored_logs/`
5. **Comparative Analysis** → Analyze automated vs human scoring patterns for research insights
6. **Research Publication** → Comprehensive methodology with dual scoring validation

### Production Command
```bash
# Run guardrail decay experiment with dual evaluation
PYTHONPATH=. python app/cli/run_experiments.py run \
  --auto-score \
  --experiment-code GRD \
  --models claude-3-7-sonnet-20250219 gpt-4.1
```

## Provenance & Status

- **Date started:** 2025-05-06
- **Current version:** v2.0.0 (streamlined with dual evaluation)
- **Status:** Active
- **Primary contributors:** Cassius Oldenburg

## Methods & Workflow

Uses systematic prompt variants to test model compliance with degraded alignment instructions across multiple conversation turns, measuring drift from baseline safety behavior.

## Results & Findings

> Results and analysis in progress with dual evaluation methodology

## Notes

- Legacy v1.0 artifacts preserved in `legacy_1.0/` directory
- Experiment uses production dual evaluation workflow
- All scored logs from iOS app are merged via PR for full auditability