# Analysis Pipeline - RED_CORE

**Phase 3 Status**: ✅ Advanced Analysis Features Complete  
**Last Updated**: 2025-06-09

## Overview

The analysis pipeline provides comprehensive evaluation and reporting capabilities for RED_CORE experiments, including automated scoring, LLM evaluation, statistical analysis, and regulatory compliance reporting.

## Key Components

### Core Evaluation System

- **`dual_evaluator.py`**: Enhanced method agreement analysis with statistical testing
- **`llm_evaluator.py`**: LLM-based semantic assessment
- **`automated_scorer.py`**: MLCommons-aligned pattern matching

### Export & Filtering

- **`batch_exporter.py`**: Enterprise CSV export with advanced filtering
- **`compliance_reporter.py`**: Regulatory compliance reporting (EU AI Act, NIST AI RMF)

## Phase 3 Features

### Enhanced Method Agreement Analysis

Statistical comparison between automated and LLM scoring methods:

```python
from app.analysis.dual_evaluator import DualEvaluator

evaluator = DualEvaluator("gemini-2.0-flash-lite")
results = await evaluator.evaluate_session_complete(session_log)

# Access enhanced comparison metrics
comparison = results["scoring_comparison"]
reliability_grade = comparison["statistical_metrics"]["reliability_grade"]  # A-F
p_value = comparison["statistical_metrics"]["p_value"]
agreement_rate = comparison["method_agreement"]["refusal_scoring"]
```

**Key Metrics:**
- Chi-square statistical significance testing
- A-F reliability grading system
- Disagreement pattern analysis
- Quality indicators (high/medium/low correlation)

### Advanced Filtering System

Multi-criteria filtering for sophisticated data analysis:

```python
from app.analysis.batch_exporter import BatchExporter

exporter = BatchExporter()

# Example 1: Filter by model and confidence
filters = {
    "model": ["gpt-4", "claude-3"],
    "min_confidence": 0.8,
    "evaluation_status": "complete"
}

# Example 2: Date range and quality filtering
filters = {
    "date_range": ("2024-01-01", "2024-12-31"),
    "max_drift": 0.3,
    "min_turns": 3
}

filtered_logs = exporter.filter_logs_advanced(logs_dir, filters)
```

**Supported Filters:**
- `batch_id`: str or List[str] - specific batch(es)
- `model`: str or List[str] - specific model(s)
- `model_vendor`: str or List[str] - vendor filtering
- `date_range`: tuple - ISO format (start_date, end_date)
- `min_confidence`/`max_drift`: float - quality thresholds
- `evaluation_status`: str - "complete", "incomplete", "failed"
- `min_turns`/`max_turns`: int - conversation length

### Regulatory Compliance Reporting

Generate standardized compliance reports for regulatory frameworks:

```python
from app.analysis.compliance_reporter import ComplianceReporter

reporter = ComplianceReporter()

# EU AI Act compliance report
ai_act_report = reporter.generate_ai_act_report(
    experiment_dirs=[Path("experiments/guardrail_decay")],
    output_path=Path("compliance/ai_act_q4_2024.json"),
    time_period="90_days"
)

# NIST AI RMF compliance report
nist_report = reporter.generate_nist_report(
    experiment_dirs=all_experiments,
    output_path=Path("compliance/nist_rmf_2024.json")
)
```

**Supported Frameworks:**

1. **EU AI Act** - Articles 9, 10, 13, 14, 15 compliance
   - Risk management system assessment
   - Data and data governance evaluation
   - Transparency and human oversight analysis
   - Accuracy, robustness and cybersecurity metrics

2. **NIST AI RMF** - Four core functions
   - GOVERN: Governance and oversight
   - MAP: Context and risk mapping
   - MEASURE: Impact assessment
   - MANAGE: Response and mitigation

**Report Features:**
- Executive summary with overall compliance scores
- Risk categorization (Critical/High/Medium/Low)
- Automated recommendations
- Audit trail generation
- Time-period filtering

## CLI Usage

### Make Commands

```bash
# Enhanced evaluation with Phase 3 features
make eval

# Interactive CSV export with filtering
make csv

# Compliance reporting
make compliance FRAMEWORK=ai_act EXPERIMENTS=exp1,exp2 OUTPUT=report.json

# Advanced filtering help
make filter
```

### Direct Python Usage

```bash
# Dual evaluation with statistical analysis
PYTHONPATH=. python -m app.analysis.dual_evaluator --smart

# Advanced CSV export with filtering
PYTHONPATH=. python -m app.analysis.batch_exporter \
  --experiment guardrail_decay \
  --model gpt-4 \
  --min-confidence 0.8

# EU AI Act compliance report
PYTHONPATH=. python -m app.analysis.compliance_reporter \
  --framework ai_act \
  --experiments guardrail_decay refusal_robustness \
  --output compliance_report.json \
  --time-period 90_days

# NIST AI RMF compliance report
PYTHONPATH=. python -m app.analysis.compliance_reporter \
  --framework nist \
  --output nist_report.json
```

## Output Examples

### Enhanced Method Agreement

```json
{
  "scoring_comparison": {
    "method_agreement": {
      "refusal_scoring": 0.847,
      "exact_agreement_count": 42,
      "total_turns": 50
    },
    "statistical_metrics": {
      "chi_square": 3.84,
      "p_value": 0.147,
      "statistically_significant": false,
      "overall_reliability": 0.912,
      "reliability_grade": "A"
    },
    "disagreement_analysis": {
      "high_disagreement_threshold": false,
      "score_disagreements": [
        {"score": 2, "auto_count": 8, "llm_count": 5, "difference": 3}
      ]
    }
  }
}
```

### EU AI Act Compliance Report

```json
{
  "executive_summary": {
    "overall_compliance_score": "A - Excellent Compliance",
    "safety_rate": "94.2%",
    "key_findings": [
      "Excellent safety performance - exceeds industry standards",
      "Evaluation coverage meets audit requirements"
    ]
  },
  "article_9_risk_management": {
    "risk_identification": {
      "methodology": "MLCommons AILuminate v1.0 + LLM evaluation",
      "assessment_coverage": "96.8%"
    }
  }
}
```

## Legacy Workflow (Phase 1-2)

### `dual_evaluator.py` - **Primary Analysis Tool** ⭐
Combined rule-based + LLM evaluation system:
- **Dual scoring methodology**: Automated patterns + LLM reasoning
- **In-place log enrichment**: Updates original logs with embedded scoring
- **Method agreement analysis**: Now with statistical testing
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
2. **Phase 3 Enhanced Dual Evaluation** → Statistical analysis + embedded scoring
3. **Advanced Filtering** → Multi-criteria data analysis
4. **Compliance Reporting** → Regulatory framework reports
5. **Human Review** → Manual validation
6. **Research Publication** → Comprehensive methodology with statistical validation

## Dependencies

Phase 3 adds:
- `scipy >= 1.11.0` for statistical testing

## Development Notes

### Error Handling

All analysis functions include comprehensive error handling:
- Malformed JSON files are skipped with warnings
- Missing evaluation data is handled gracefully  
- Statistical tests require minimum sample sizes
- Compliance reports handle missing data appropriately

### Performance Considerations

- Advanced filtering operates in-memory for speed
- Statistical tests only run on sufficient sample sizes (n≥10)
- Compliance reporting can handle large experiment sets
- Chi-square tests use optimized numpy operations

---

For detailed API documentation, see `CLAUDE.md` in this directory.