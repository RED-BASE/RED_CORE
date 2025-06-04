#!/bin/bash

# Set environment variables
export LOG_DIR=experiments/refusal_robustness/logs/
export PYTHONPATH=.

# Run all 8 refusal robustness experiments
echo "Running refusal robustness experiment 01..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_01_direct_ask.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 02..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_02_fiction_author.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 03..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_03_distressed_user.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 04..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_04_investigative_journalist.yaml \
  --experiment-code RRS

echo "Running refusal robustness experiment 05..."
poetry run python app/cli/run_experiments.py run \
  --models gpt-4o claude-3-7-sonnet-20250219 gemini-pro \
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml \
  --usr-prompt data/prompts/user/refusal/refusal_robustness_misinformation_tweet_05_satirical_pundit.yaml \
  --experiment-code RRS

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