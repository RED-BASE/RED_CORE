LOG_DIR=experiments/80k_hours_demo/logs/ \
PYTHONPATH=. poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/80k_hours_demo/usr_80k_hours_demo_01.yaml \
  --experiment-code 80K
