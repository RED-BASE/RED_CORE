# RED_CORE Makefile

# Create a development log entry
log:
	bash dev_log/reflections/create_log_entry.sh

# Run experiments interactively
run:
	PYTHONPATH=. python app/cli/run_experiments.py run --interactive

# Run tests
test:
	PYTHONPATH=. python -m pytest tests/

# Help
help:
	@echo "Available targets:"
	@echo "  run        - Run experiments interactively (dual evaluation enabled by default)"
	@echo "  test       - Run test suite"
	@echo "  log        - Create development log entry"
	@echo "  help       - Show this help"