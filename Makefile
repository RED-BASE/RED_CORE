# RED_CORE Makefile
# Adversarial AI Safety Research Framework

.PHONY: help exp usr sys run eval csv status test clean compliance filter rates


# Run experiments interactively
run:
	poetry run python -m app.cli.run_experiments run --interactive

# PHASE 1: Enhanced workflow commands
exp:
	@echo "🚀 Interactive Experiment Creator"
	@if [ -n "$(AI)" ]; then \
		PYTHONPATH=. python -m app.cli.experiment_creator --ai-model $(AI); \
	else \
		PYTHONPATH=. python -m app.cli.experiment_creator; \
	fi

usr:
	@echo "📝 Interactive User Prompt Builder"
	PYTHONPATH=. python -m app.cli.prompt_builder

sys:
	@echo "🎭 Interactive System Prompt Builder"
	PYTHONPATH=. python -m app.cli.prompt_builder system

eval:
	@echo "🔍 Running Smart Evaluation (incomplete logs only)..."
	PYTHONPATH=. python -m app.analysis.dual_evaluator --smart

csv:
	@echo "📊 Interactive CSV Export Menu"
	PYTHONPATH=. python -m app.analysis.batch_exporter --interactive

status:
	@echo "📋 Experiment Status Dashboard"
	PYTHONPATH=. python -m app.cli.experiment_dashboard

# PHASE 3: Advanced Analysis Commands
compliance:
	@echo "📋 Regulatory Compliance Reporting"
	@echo "Available frameworks: ai_act, nist"
	@echo "Usage: make compliance FRAMEWORK=ai_act [EXPERIMENTS=exp1,exp2] [OUTPUT=report.json]"
	@if [ -n "$(FRAMEWORK)" ]; then \
		PYTHONPATH=. python -m app.analysis.compliance_reporter --framework $(FRAMEWORK) \
			$$(if [ -n "$(EXPERIMENTS)" ]; then echo "--experiments $$(echo $(EXPERIMENTS) | tr ',' ' ')"; fi) \
			--output $${OUTPUT:-compliance_report_$(FRAMEWORK).json}; \
	else \
		echo "Please specify FRAMEWORK (ai_act or nist)"; \
	fi

filter:
	@echo "🔍 Advanced Log Filtering"
	@echo "Usage examples:"
	@echo "  python -m app.analysis.batch_exporter --experiment EXP --model 'gpt-4' --min-confidence 0.8"
	@echo "  python -m app.analysis.batch_exporter --experiment EXP --date-range '2024-01-01' '2024-12-31'"
	@echo "Run with --help for full options"


# Rate limit monitoring
rates:
	@echo "⚡ Rate Limit Status Monitor"
	@if [ "$(filter watch,$(MAKECMDGOALS))" = "watch" ]; then \
		PYTHONPATH=. python -m app.cli.rate_limit_status --watch; \
	else \
		PYTHONPATH=. python -m app.cli.rate_limit_status; \
	fi

# Support for 'make rates watch'
watch:
	@true

# Run tests
test:
	PYTHONPATH=. python -m pytest tests/

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Help
help:
	@echo "🔴 RED_CORE - AI Safety Framework"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🏠 Navigation:"
	@echo "  exit           - Return to GitHub root"
	@echo ""
	@echo "🚀 Core Workflow:"
	@echo "  exp            - Create new experiment"
	@echo "  usr            - Build user prompts"
	@echo "  sys            - Build system prompts"
	@echo "  run            - Run experiments"
	@echo "  eval           - Evaluate results"
	@echo "  csv            - Export to CSV"
	@echo "  status         - Experiment dashboard"
	@echo ""
	@echo "📊 PHASE 3 - Advanced Analysis:"
	@echo "  compliance     - Generate regulatory reports"
	@echo "  filter         - Advanced log filtering help"
	@echo ""
	@echo "⚡ Performance & Monitoring:"
	@echo "  rates          - View rate limit status"
	@echo "  rates watch    - Monitor rate limits live"
	@echo ""
	@echo "🧹 Utilities:"
	@echo "  test           - Run test suite"
	@echo "  clean          - Remove Python cache files"