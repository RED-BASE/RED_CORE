# data/prompts/system/

Global system prompts used across multiple experiments.

## Available Prompts

- `sys_helpful_assistant.yaml` - Standard helpful AI assistant behavior
- `sys_unhelpful_assistant.yaml` - Deliberately unhelpful responses for comparison  
- `sys_none_00.yaml` - Minimal/no system prompt baseline

## Usage Note

These are **global** system prompts available to all experiments. Experiment-specific system prompts are located in their respective `experiments/{experiment_name}/prompts/` directories.

The interactive runner (`make run`) will discover and present both global and experiment-specific system prompts automatically.