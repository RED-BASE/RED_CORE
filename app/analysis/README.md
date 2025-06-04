# analysis/

Scripts and tools for log analysis, summarization, and review.

## 🎯 Quick Start

Use the unified analysis CLI for all analysis tasks:

```bash
# Run complete analysis for an experiment
poetry run python app/analysis/analysis_cli.py full-analysis --experiment refusal_robustness

# Quick insights from existing flat logs
poetry run python app/analysis/analysis_cli.py quick-insights

# Extract metadata from raw logs  
poetry run python app/analysis/analysis_cli.py extract-metadata --log-dir experiments/refusal_robustness/logs/

# Interactive review tool
poetry run python app/analysis/analysis_cli.py review --log-dir experiments/refusal_robustness/logs/
```

## 📁 Tools Overview

### `analysis_cli.py` - **Unified Interface** 
Coordinates all analysis tools through a single command-line interface. **Use this for all analysis tasks.**

### `log_insight_report.py` - **Quick Analytics**
Fast statistical overview of experiment results:
- Model performance comparisons
- Drift pattern analysis  
- Anomaly detection

### `analyze.py` - **Metadata Extraction**
Extracts structured metadata from raw experiment logs:
- Run summaries and statistics
- Alignment pattern detection
- System prompt drift analysis

### `roller.py` - **Data Aggregation**
Rolls up turn-level data into run-level and experiment-level summaries:
- Trajectory classification (stable, escalating, snap, backslide)
- Aggregate statistics
- Cross-experiment comparisons

### `enhanced_review_tool.py` - **Interactive Review**
Comprehensive manual review interface:
- Color-coded turn-by-turn analysis
- Interactive scoring and annotation
- Export capabilities for scored data

## 🔄 Analysis Workflow

1. **Run Experiments** → Raw JSON logs saved to `experiments/{name}/logs/`
2. **Extract Metadata** → Structured summaries and statistics  
3. **Flatten Data** → Turn-level CSV for detailed analysis
4. **Roll Up Data** → Run-level and experiment-level aggregations
5. **Generate Insights** → Quick statistical reports
6. **Manual Review** → Interactive scoring via enhanced review tool
7. **Export Results** → Scored logs to `experiments/{name}/scored_logs/`

## 📊 Data Flow

```
Raw Logs (JSON) 
    ↓ extract-metadata
Summary Data (CSV)
    ↓ flatten (via core/flattener.py)  
Flat Logs (CSV)
    ↓ roll-up
Rolled Data (CSV)
    ↓ quick-insights + review
Analysis Reports + Scored Logs
``` 

