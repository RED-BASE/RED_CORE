
# LLM-REDCORE Makefile

# Create a log entry
log:
	bash -m dev_log/reflections/create_log_entry.sh


.PHONY: review-logs review-logs-keywords

old_review:
	python review_assistant.py

# ... (other Makefile content, if any) ...

# Define the log directory relative to the Makefile's location
LOG_DIR ?= logs/raw

review:
	python enhanced_review_tool.py $(LOG_DIR)

# You can also add options to the python command if you want defaults:
# review:
#	python enhanced_review_tool.py $(LOG_DIR) --auto-pass --keywords "harmful,unsafe"

# run experiments interactively
run:
	PYTHONPATH=. poetry run python app/cli/run_experiments.py run --interactive
