#!/bin/bash

# Set environment variables
export LOG_DIR=experiments/refusal_robustness/logs/
export PYTHONPATH=.

# Run refusal robustness experiments
echo "Running refusal robustness experiment 06..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_06_academic_researcher.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 07..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_07_whistleblower_sim.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 08..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_08_teen_viral.yaml \
  --experiment-code RRS

echo "All experiments complete!"