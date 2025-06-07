# RED_CORE Makefile

# Create a development log entry
log:
	bash dev_log/reflections/create_log_entry.sh

# Run experiments interactively
run:
	PYTHONPATH=. python app/cli/run_experiments.py run --interactive

# Run experiments with auto-scoring (production workflow)
run-auto:
	PYTHONPATH=. python app/cli/run_experiments.py run --interactive --auto-score

# Run dual evaluation on existing logs
evaluate:
	PYTHONPATH=. python -m app.analysis.dual_evaluator --log-dir $(LOG_DIR)

# Run automated scoring only
score:
	PYTHONPATH=. python -m app.analysis.automated_scorer --log-dir $(LOG_DIR)

# Run LLM evaluation only  
llm-eval:
	PYTHONPATH=. python -m app.analysis.llm_evaluator --log-dir $(LOG_DIR)

# Run tests
test:
	PYTHONPATH=. python -m pytest tests/

# Help
help:
	@echo "Available targets:"
	@echo "  run        - Run experiments interactively"
	@echo "  run-auto   - Run experiments with auto-scoring"
	@echo "  evaluate   - Run dual evaluation on LOG_DIR"
	@echo "  score      - Run automated scoring on LOG_DIR"
	@echo "  llm-eval   - Run LLM evaluation on LOG_DIR"
	@echo "  test       - Run test suite"
	@echo "  log        - Create development log entry"
	@echo "  help       - Show this help"