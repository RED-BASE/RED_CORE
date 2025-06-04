# RED_CORE

A comprehensive framework for adversarial AI safety research, focusing on red team attacks, refusal robustness, and guardrail evaluation.

## 🎯 Overview

RED_CORE is designed for systematic exploration of AI safety boundaries through:
- **Refusal Robustness Testing**: Multi-persona attacks against content policies
- **Guardrail Decay Analysis**: Progressive degradation of safety mechanisms  
- **Attack Pattern Discovery**: Systematic cataloging of adversarial techniques
- **Reproducible Research**: Strict provenance tracking and auditable experiments

## 🏗️ Architecture

```
RED_CORE/
├── app/                    # Core application logic
│   ├── analysis/          # Log analysis and scoring tools
│   ├── api_runners/       # Model API interfaces (OpenAI, Anthropic, Google)
│   ├── cli/               # Command-line experiment orchestration
│   └── core/              # Shared utilities and data schemas
├── data/                  # Experimental configurations
│   ├── prompts/           # System and user prompt templates
│   ├── personas/          # Attack persona definitions
│   └── flattened/         # Processed experiment data
├── experiments/           # Experiment results and configurations
├── safety/                # Safety containment and filtering
└── tools/                 # Development and enforcement scripts
```

## 🚀 Quick Start

### Installation
```bash
# Install dependencies (never use pip directly)
poetry install --no-root

# Verify installation
poetry run python app/cli/run_experiments.py --help
```

### Running Your First Experiment
```bash
# Run a refusal robustness test
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_01_direct_ask.yaml \
  --experiment-code RRS
```

### Create New Experiments
```bash
poetry run run-experiments new \
  --name my_experiment \
  --contributors "researcher@example.com" \
  --purpose "Testing new attack vector"
```

## 🧪 Experiment Types

### Refusal Robustness (`RRS`)
Tests model resistance to harmful content generation across 8 attack personas:
- Direct requests
- Role-playing scenarios (journalist, academic, fiction writer)
- Social engineering (distressed user, whistleblower simulation)
- Viral content creation

### Guardrail Decay (`GRD`) 
Evaluates progressive weakening of safety mechanisms through:
- Iterative prompt modifications
- Context window exploitation
- Safety system fatigue testing

## 📊 Analysis Workflow

1. **Raw Logs**: Saved to `experiments/{experiment_name}/logs/`
2. **Analysis**: Use tools in `app/analysis/` for insights
3. **Scoring**: Manual review via iOS app → `scored_logs/`
4. **Aggregation**: Automated summaries and trend analysis

```bash
# Generate insights from recent experiments
poetry run python app/analysis/log_insight_report.py

# Create comprehensive analysis
poetry run python app/analysis/enhanced_review_tool.py
```

## 🛡️ Safety & Security

### Content Filtering
- Automatic content containment in `safety/containment.py`
- Multi-layer filtering for harmful outputs
- Secure API key management

### Provenance Tracking
- All file operations audited and restricted
- Complete experiment reproducibility
- Git hooks enforce data integrity

## 🔧 Development Rules

### Dependency Management
```bash
# ✅ ALWAYS use Poetry
poetry add <package>

# ❌ NEVER use pip directly  
pip install <anything>  # Rejected by pre-commit hooks
```

### File I/O Restrictions
Write operations are **blocked** in all scripts except:
- `app/cli/run_experiments.py` (orchestrator)
- `meta/friction_audit.py` (meta-analysis)
- `dev_log/reflections/create_log_entry.sh` (development logs)

**Rationale**: Centralized data mutation ensures reproducibility and auditability.

### Code Quality
- All tool usage via `poetry run <command>`
- Structured logging (no `print()` debugging)
- Type hints and Pydantic validation
- Pre-commit hooks for consistent formatting

## 📈 Model Support

Currently supported APIs:
- **OpenAI**: GPT-4, GPT-3.5-turbo variants
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus)  
- **Google**: Gemini Pro, Gemini Pro Vision
- **Local**: llama.cpp integration

Add new models via `app/config/config.py` model registry.

## 🤝 Contributing

1. **Experiment Creation**: Use `poetry run run-experiments new`
2. **Code Changes**: Follow pre-commit hook requirements
3. **Documentation**: Update relevant READMEs for new features
4. **Safety Review**: All outputs manually reviewed before merge

## 📝 Logging

Each experiment produces structured JSON logs containing:
- Per-turn model responses and metadata
- Token usage and latency metrics  
- Drift scores and safety annotations
- Complete prompt provenance chain

## ⚠️ Research Ethics

This framework is designed for:
- ✅ Academic AI safety research
- ✅ Responsible disclosure of vulnerabilities
- ✅ Improving model robustness

**Not intended for**:
- ❌ Malicious use or exploitation
- ❌ Production system attacks
- ❌ Uncontrolled model jailbreaking

All research should follow responsible disclosure practices and institutional review guidelines.

---

For detailed usage instructions, see individual README files in each component directory.
