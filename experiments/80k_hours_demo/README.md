# 80K Hours Demo Experiment

**Experiment Code**: `80K`  
**Purpose**: Demonstration of RED_CORE capabilities for 80,000 Hours research collaboration  
**Status**: Ready for external review

## 🎯 Objective

This experiment serves as a proof-of-concept demonstration of RED_CORE's AI safety research capabilities, specifically designed to showcase:

- Multi-model comparative analysis across leading AI systems
- Systematic refusal robustness testing methodology  
- Dual evaluation system (rule-based + LLM scoring)
- Research-grade logging with complete provenance
- Reproducible experiment design

## 🧪 Experimental Design

### Models Tested
- **GPT-4.1** (OpenAI flagship with 1M context)
- **Claude Opus 4** (Anthropic's most capable model)  
- **Gemini 2.5 Pro** (Google's latest with thinking capability)

### Safety Scenario
Progressive refusal testing using a standardized helpful assistant baseline with carefully designed prompts that test safety boundary responses.

### Methodology
1. **Baseline Establishment**: Standard helpful assistant system prompt
2. **Prompt Delivery**: Systematic user prompts designed to elicit safety responses
3. **Response Collection**: Full conversation logging with metadata
4. **Dual Evaluation**: Automated rule-based scoring + LLM-based assessment
5. **Analysis**: Statistical comparison across models and safety patterns

## 📊 Expected Outputs

### Raw Data
- Complete conversation logs in `logs/` directory
- JSON format with embedded methodology and provenance
- Token usage, latency, and API metadata

### Analysis Results  
- Refusal rates by model and prompt type
- Safety mechanism effectiveness comparison
- Detailed reasoning from LLM evaluator
- Confidence metrics for all assessments

## 🚀 Running the Experiment

### Quick Start
```bash
# Interactive mode (recommended)
make run
# Select: 80k_hours_demo experiment, choose models, run

# Direct command
LOG_DIR=experiments/80k_hours_demo/logs/ \
PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
  --models gpt-4.1 claude-opus-4-20250514 gemini-2.5-pro-preview-06-05 \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt experiments/80k_hours_demo/prompts/usr_80k_hours_demo_01.yaml \
  --experiment-code 80K
```

### Post-Experiment Analysis
```bash
# Run dual evaluation
PYTHONPATH=. poetry run python app/analysis/dual_evaluator.py \
  --log-dir experiments/80k_hours_demo/logs/

# Interactive review
PYTHONPATH=. poetry run python app/analysis/enhanced_review_tool.py \
  experiments/80k_hours_demo/logs/
```

## 📋 Research Context

This experiment demonstrates RED_CORE's suitability for:

- **Academic AI Safety Research**: Rigorous methodology with full reproducibility
- **Cross-Model Comparative Studies**: Standardized evaluation across vendors
- **Policy Research**: Evidence-based analysis of safety mechanism effectiveness
- **Responsible Disclosure**: Systematic identification of potential vulnerabilities

## 🔬 Technical Innovation

### Schema Design
- Research-ready JSON logs with embedded methodology
- Complete provenance chain for reproducibility
- Dual scoring system for robust evaluation

### Evaluation Framework
- Rule-based scoring for consistency and speed
- LLM-based assessment for nuanced analysis  
- Method agreement analysis for validation
- Confidence metrics for reliability assessment

## 🛡️ Safety Considerations

- All experiments conducted within ethical research guidelines
- No malicious exploitation or production system attacks
- Results intended for improving AI safety, not circumventing it
- Manual review required before any data sharing

## 📬 Contact

For questions about this experiment or collaboration opportunities:
- **Project Lead**: Cassius Oldenburg
- **Framework**: RED_CORE AI Safety Research Platform
- **Repository**: Private (external access by invitation)

---

*This experiment showcases RED_CORE's capabilities as a production-ready framework for systematic AI safety research.*