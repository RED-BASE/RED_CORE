
# LLM-REDCORE Makefile

# Create a log entry
log:
	bash -m dev_log/reflections/create_log_entry.sh

# Run the review tool
review:
	python enhanced_review_tool.py $(LOG_DIR)

# run experiments interactively
run:
	PYTHONPATH=. poetry run python app/cli/run_experiments.py run --interactive
