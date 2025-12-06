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
│   └── personas/          # Attack persona definitions
├── experiments/           # Experiment results and configurations
└── tools/                 # Development scripts and containment
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
# Interactive mode (recommended) - guides you through experiment setup
make run

# Or run a specific refusal robustness test
PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
  --models gpt-4.1 claude-opus-4-20250514 \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt experiments/refusal_robustness/prompts/refusal_robustness_misinformation_tweet_01_direct_ask.yaml \
  --experiment-code RRS
```

**Note**: User prompts live in each experiment's `prompts/` folder (e.g., `experiments/refusal_robustness/prompts/`). System prompts and personas are shared in `data/prompts/`.

### Quick Setup
```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see API Key Setup section below)

# Test installation with sample experiment
make run
```

### API Key Setup

RED_CORE supports 8 AI providers. Set up the API keys you need:

```bash
# Required for basic functionality
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key" 
export GOOGLE_API_KEY="your_google_key"

# Optional: Additional providers for diverse perspectives
export XAI_API_KEY="your_xai_key"              # Get at: https://console.x.ai/
export DEEPSEEK_API_KEY="your_deepseek_key"    # Get at: https://platform.deepseek.com/
export MISTRAL_API_KEY="your_mistral_key"      # Get at: https://console.mistral.ai/
export COHERE_API_KEY="your_cohere_key"        # Get at: https://dashboard.cohere.ai/
export TOGETHER_API_KEY="your_together_key"    # Get at: https://api.together.xyz/
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
3. **Human Review**: Manual validation and edge case analysis
4. **Comparative Analysis**: Method agreement analysis between automated and human scoring

### Automated Scoring Strategy (Updated 2025-06-08)
- **Current**: MLCommons-inspired regex patterns (frozen as 2024 baseline)
- **Maintenance**: Annual updates aligned with MLCommons official releases
- **Next Update**: MLCommons v1.0 (expected late 2024/early 2025)
- **Future**: Transition from regex to Llama Guard methodology
- **Rationale**: Balance institutional credibility with manageable maintenance

```bash
# Run dual evaluation (rule-based + LLM scoring)
PYTHONPATH=. python -m app.analysis.dual_evaluator --log-dir experiments/refusal_robustness/logs/

# Or use integrated experiment runner (dual evaluation enabled by default)
PYTHONPATH=. python app/cli/run_experiments.py run --experiment-code RRS --models claude-3-7-sonnet-20250219

# Individual scoring components
PYTHONPATH=. python -m app.analysis.automated_scorer --log-dir experiments/refusal_robustness/logs/
PYTHONPATH=. python -m app.analysis.llm_evaluator --log-dir experiments/refusal_robustness/logs/
```

## 🛡️ Safety & Security

### Content Filtering
- Automatic content containment in `tools/containment.py`
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

**Rationale**: Centralized data mutation ensures reproducibility and auditability.

### Code Quality
- All tool usage via `poetry run <command>`
- Structured logging (no `print()` debugging)
- Type hints and Pydantic validation
- Pre-commit hooks for consistent formatting

## 📈 Model Support

Currently supported APIs (27+ models across 8 providers):
- **OpenAI**: GPT-4.1 series (flagship with 1M context), GPT-4o variants
- **Anthropic**: Claude 4 (Opus, Sonnet), Claude 3.7 Sonnet, Claude 3.5 series  
- **Google**: Gemini 2.5 (Pro, Flash with thinking), Gemini 2.0 Flash, Gemini 1.5 series
- **XAI**: Grok (uncensored approach with real-time data access)
- **DeepSeek**: Chat and Coder models (strong Chinese AI perspective, excellent coding)
- **Mistral AI**: Large, Small, and Codestral models (European approach, specialized coding)
- **Cohere**: Command R+ and Command R (enterprise RAG capabilities)
- **Together AI**: Llama 3.3 70B, Llama 3.2 Vision (latest open models)
- **Local**: Mistral 7B via llama.cpp integration

### Diverse Perspective Coverage
- **Most Safety-Conscious**: Claude models with strong refusal patterns
- **Most Uncensored**: Grok for edge case testing  
- **Best Coding**: DeepSeek Coder + Codestral for technical attacks
- **Largest Context**: Gemini 2.5 Pro (2M tokens) for complex scenarios
- **Enterprise Focus**: Cohere for business use case testing

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
