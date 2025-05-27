
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

# ... (other Makefile content, if any) ...
fric:
	python meta/friction_audit.py


run:
	poetry run python -m app.cli.run_experiments run \
	  --models gpt-4o \
	  --sys-prompt data/prompts/system/sys_none_00.yaml \
	  --usr-prompt data/prompts/user/guardrail_decay/usr_test.yaml
