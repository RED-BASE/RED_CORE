# Sarah Sails System Prompts Experiment

**Experiment Code**: `SAR`  
**Purpose**: Test system prompts for AI companion "Sarah Sails"  
**Status**: In Development

## 🎯 Objective

This experiment assesses robustness of system prompts for the Sarah Sails AI companion in realistic simulated deployments.

## 🧪 Experimental Design

### Research Question
Will the system prompt help the model generate in character responses consistently?

### Models to Test
- **GPT-4.1** (OpenAI flagship)\n- **Claude Opus 4** (Anthropic)\n- **Gemini 2.5 Pro** (Google)

### Safety Scenario
Describe the specific safety scenario being tested.

### Methodology
1. **Baseline Establishment**: System prompt configuration
2. **Prompt Delivery**: User prompts designed for this experiment
3. **Response Collection**: Complete conversation logging
4. **Dual Evaluation**: Automated + LLM-based assessment
5. **Analysis**: Statistical analysis and safety pattern identification

## 📊 Expected Outputs

### Raw Data
- Complete conversation logs in `logs/` directory
- JSON format with methodology and provenance
- Token usage, latency, and API metadata

### Analysis Results  
- Key metrics and findings expected from this experiment
- Safety mechanism effectiveness analysis
- Detailed reasoning from evaluator systems
- Confidence metrics and reliability assessment

## 🚀 Running the Experiment

### Quick Start
```bash
# Interactive mode (recommended)
make run
# Select: sarah_sails_system_prompts experiment

# Direct command
PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
  --models gpt-4.1 claude-opus-4-20250514 \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt experiments/sarah_sails_system_prompts/prompts/usr_sarah_sails_system_prompts_[prompt_name].yaml \
  --experiment-code SAR
```

### Post-Experiment Analysis
```bash
# Smart evaluation (only incomplete logs)
make eval

# Interactive CSV export
make csv
```

## 📋 Research Context

This experiment is designed to investigate:

- Primary research goal
- Secondary research goal
- Tertiary research goal

## 🔬 Technical Innovation

### Schema Design
- Research-ready JSON logs with embedded methodology
- Complete provenance chain for reproducibility
- Batch tracking for iterative development

### Evaluation Framework
- Rule-based scoring for consistency
- LLM-based assessment for nuanced analysis  
- Method agreement analysis for validation

## 🛡️ Safety Considerations

- All experiments conducted within ethical research guidelines
- No malicious exploitation or production system attacks
- Results intended for improving AI safety
- Manual review required before data sharing

## 📊 Batch Tracking

This experiment uses RED_CORE's batch tracking system:
- Each run gets a unique batch ID (e.g., `sarah_sails_system_prompts-01`)
- CSV exports available per batch for analysis
- Workflow state tracking prevents duplicate evaluation

---

*Created with RED_CORE Experiment Creator - 2026-01-01*
