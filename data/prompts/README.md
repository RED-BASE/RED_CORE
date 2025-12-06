# data/prompts/

Global prompt templates and personas shared across experiments.

## Structure

- **system/** - Global system prompts (helpful, unhelpful, baseline)
- **user/** - User prompt templates and shared prompts  
- **personas/** - User personas for adversarial testing (malicious_actor, etc.)

## Usage Philosophy

**Global vs Experiment-Specific:**
- **Global prompts** (in `data/prompts/`) are reusable across multiple experiments
- **Experiment-specific prompts** live in `experiments/{experiment_name}/prompts/`

The interactive runner automatically discovers both global and experiment-specific prompts, presenting them as unified options while maintaining logical organization.