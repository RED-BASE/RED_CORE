# RED_CORE

A framework for adversarial AI safety research, focusing on red team attacks, refusal robustness, and guardrail evaluation.

## Overview

RED_CORE enables systematic exploration of AI safety boundaries through:
- **Refusal Robustness Testing**: Multi-persona attacks against content policies
- **Guardrail Decay Analysis**: Progressive degradation of safety mechanisms
- **Attack Pattern Discovery**: Systematic cataloging of adversarial techniques
- **Reproducible Research**: Strict provenance tracking and auditable experiments

## Architecture

```
RED_CORE/
├── app/
│   ├── analysis/          # Scoring and evaluation tools
│   ├── api_runners/       # Model API interfaces (8 providers)
│   ├── cli/               # Interactive experiment orchestration
│   └── core/              # Schemas, logging, rate limiting
├── data/
│   └── prompts/           # System prompts and personas
├── experiments/           # Per-experiment prompts, logs, configs
└── tools/                 # Containment and hooks
```

## Quick Start

```bash
# Install
poetry install --no-root
cp .env.example .env  # Add your API keys

# Run interactively
make run
```

The interactive CLI guides you through:
1. Select experiment → auto-generates experiment code
2. Select system prompts (multi-select)
3. Select user prompts (filtered to experiment)
4. Select models (multi-select)
5. Optional: personas
6. Run

## Experiments

### Refusal Robustness
Tests model resistance to harmful content generation across 8 attack personas:
- Direct requests
- Role-playing scenarios (journalist, academic, fiction writer)
- Social engineering (distressed user, whistleblower simulation)
- Viral content creation

### Guardrail Decay
Evaluates progressive weakening of safety mechanisms through:
- Iterative prompt modifications
- Context window exploitation
- Safety system fatigue testing

### Unicode Homoglyphs
Tests filter bypass via character substitution:
- Cyrillic/Latin swaps
- Mathematical symbols
- Lookalike characters

## Analysis

Dual evaluation system:
1. **Rule-based**: MLCommons-inspired regex patterns
2. **LLM-based**: Model-as-judge assessment

```bash
make eval      # Smart evaluation (incomplete logs only)
make csv       # Export results
make status    # Dashboard view
```

## Model Support

27+ models across 8 providers:

| Provider | Models | Notes |
|----------|--------|-------|
| OpenAI | GPT-4.1, GPT-4o | 1M context flagship |
| Anthropic | Claude 4, Claude 3.7/3.5 | Strong safety patterns |
| Google | Gemini 2.5/2.0 | 2M context, thinking modes |
| xAI | Grok | Minimal filtering |
| DeepSeek | Chat, Coder | Chinese perspective |
| Mistral | Large, Codestral | European approach |
| Cohere | Command R+ | Enterprise RAG |
| Together | Llama 3.3 | Open models |

## CLI Commands

```bash
make run       # Run experiments (interactive)
make exp       # Create new experiment
make usr       # Build user prompts
make sys       # Build system prompts
make eval      # Evaluate results
make csv       # Export to CSV
make status    # Experiment dashboard
make rates     # Rate limit monitor
```

## Development

```bash
poetry add <package>    # Always use Poetry
make test               # Run test suite
```

File I/O is restricted to the orchestrator (`run_experiments.py`) for reproducibility.

## Research Ethics

Designed for:
- Academic AI safety research
- Responsible vulnerability disclosure
- Model robustness improvement

Not intended for malicious exploitation or production attacks.

---

See `data/model_registry.md` for model specifications and individual `CLAUDE.md` files for component documentation.
