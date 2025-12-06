# experiments/guardrail_decay/prompts/

Experiment-specific prompts for guardrail decay testing.

## System Prompts

- `sys_guardrail_decay_01.yaml` - Initial guardrail degradation instructions
- `sys_guardrail_decay_02.yaml` - Moderate guardrail decay simulation  
- `sys_guardrail_decay_03.yaml` - Severe guardrail decay conditions

## User Prompts

- `usr_guardrail_decay_01.yaml` - Primary guardrail decay test prompts
- `usr_test.yaml` - Test/development prompt variants

## Usage

These prompts are automatically discovered by the interactive runner when the guardrail_decay experiment is selected. The system also has access to global system prompts from `data/prompts/system/` for baseline comparisons.