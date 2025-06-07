# analysis/

Scripts and tools for log analysis, evaluation, and review.

## 🎯 Quick Start

Use the dual evaluator for comprehensive automated analysis:

```bash
# Run dual evaluation (automated + LLM scoring) on experiment logs
PYTHONPATH=. python -m app.analysis.dual_evaluator --log-dir experiments/refusal_robustness/logs/

# Run dual evaluation with custom LLM evaluator model
PYTHONPATH=. python -m app.analysis.dual_evaluator --log-dir experiments/refusal_robustness/logs/ --evaluator-model gemini-2.0-flash-lite

# Run automated scoring only
PYTHONPATH=. python -m app.analysis.automated_scorer --log-dir experiments/refusal_robustness/logs/ --standard red_core_custom

# Run LLM evaluation only
PYTHONPATH=. python -m app.analysis.llm_evaluator --log-dir experiments/refusal_robustness/logs/ --evaluator-model claude-3-7-sonnet

# Or use the integrated --auto-score flag in run_experiments.py (recommended)
PYTHONPATH=. python app/cli/run_experiments.py run --auto-score --experiment-code RRS --models claude-3-7-sonnet-20250219
```

## 📁 Tools Overview

### `dual_evaluator.py` - **Primary Analysis Tool** ⭐
Combined rule-based + LLM evaluation system:
- **Dual scoring methodology**: Automated patterns + LLM reasoning
- **In-place log enrichment**: Updates original logs with embedded scoring
- **Method agreement analysis**: Compares scoring approaches for validation
- **Cost-optimized**: Default gemini-2.0-flash-lite evaluator (~90% cost savings)
- **Research-grade output**: Publication-ready logs with complete provenance

### `automated_scorer.py` - **Rule-Based Evaluation**
Industry-standard automated safety scoring:
- **MLCommons AILuminate** patterns and RED_CORE custom rules
- Multi-pattern refusal detection (1=hard refusal, 2=partial, 3=full compliance)
- Hazard category classification (12+ categories)
- Fast, deterministic scoring with confidence metrics

### `llm_evaluator.py` - **LLM-Based Evaluation**
Nuanced contextual evaluation using language models:
- **Model-agnostic**: Supports Claude, GPT, and Gemini evaluators
- Contextual assessment with detailed reasoning
- Confidence scoring and drift analysis
- Async processing with rate limiting

## 🔄 Analysis Workflow

1. **Run Experiments** → Raw JSON logs saved to `experiments/{name}/logs/`
2. **Automated Dual Evaluation** → Rule-based + LLM scoring embedded in logs
3. **Method Agreement Analysis** → Compare automated vs LLM scoring approaches
4. **iOS red_score App Review** → Human validation and scoring via mobile interface
5. **Comparative Analysis** → Analyze automated vs human scoring patterns for research insights
6. **Research Publication** → Comprehensive methodology with dual scoring validation

## 📊 Data Flow

```
Raw Logs (JSON)
    ↓ dual_evaluator.py
Enriched Logs (JSON) + Analysis Reports
    ↓ red_score iOS app
Human-Scored Logs (JSON)
    ↓ comparative analysis
Research-Ready Dataset with Dual Methodology
```

## 🎯 Production Integration

The analysis pipeline integrates seamlessly with experiment runs:

```bash
# Single command: experiments → scoring → ready for human review
PYTHONPATH=. python app/cli/run_experiments.py run \
  --auto-score \
  --experiment-code RRS \
  --models claude-3-7-sonnet-20250219 gpt-4.1
```

Output: Fully scored logs ready for iOS app human validation and research publication.