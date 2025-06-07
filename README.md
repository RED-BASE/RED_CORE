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
# Interactive mode (recommended)
make run

# Or run a specific refusal robustness test
PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
  --models gpt-4.1 claude-opus-4-20250514 \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_01_direct_ask.yaml \
  --experiment-code RRS
```

### Quick Setup
```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, Google)

# Test installation with sample experiment
make run
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
2. **Dual Evaluation**: Automated rule-based + LLM scoring (in-place enrichment)
3. **Human Review**: iOS red_score app for validation and edge case analysis
4. **Comparative Analysis**: Method agreement analysis between automated and human scoring

```bash
# Run dual evaluation (rule-based + LLM scoring)
PYTHONPATH=. python -m app.analysis.dual_evaluator --log-dir experiments/refusal_robustness/logs/

# Or use integrated --auto-score flag (recommended)
PYTHONPATH=. python app/cli/run_experiments.py run --auto-score --experiment-code RRS --models claude-3-7-sonnet-20250219

# Individual scoring components
PYTHONPATH=. python -m app.analysis.automated_scorer --log-dir experiments/refusal_robustness/logs/
PYTHONPATH=. python -m app.analysis.llm_evaluator --log-dir experiments/refusal_robustness/logs/
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

Currently supported APIs (16 models available):
- **OpenAI**: GPT-4.1 series (flagship with 1M context), GPT-4o variants
- **Anthropic**: Claude 4 (Opus, Sonnet), Claude 3.7 Sonnet, Claude 3.5 series  
- **Google**: Gemini 2.5 (Pro, Flash with thinking), Gemini 2.0 Flash, Gemini 1.5 series
- **Local**: Mistral 7B via llama.cpp integration

All models updated to latest SDK versions (June 2025). See `data/model_registry.md` for complete specifications.

## 🤝 Contributing

1. **Experiment Creation**: Use `poetry run run-experiments new`
2. **Code Changes**: Follow pre-commit hook requirements
3. **Documentation**: Update relevant READMEs for new features
4. **Safety Review**: All outputs manually reviewed before merge

## 📝 Logging

Each experiment produces research-ready structured JSON logs containing:
- Complete conversation turns with model responses
- Full system prompt content and file references  
- Token usage, latency, and API metadata
- Dual evaluation scores (rule-based + LLM assessment)
- Provenance chain for full reproducibility
- Embedded experiment methodology and parameters

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
