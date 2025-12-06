# data/

Project data, documentation, and prompt artifacts.

- **Claude.md**: Development context and architecture documentation
- **model_registry.md**: Comprehensive model specifications and capabilities
- **prompts/**: Prompt templates and personas for experiments
  - **system/**: System prompt YAMLs for different experimental conditions
  - **user/**: User prompt template (experiment-specific prompts moved to experiments/ directories)
  - **personas/**: YAML files defining user personas for adversarial testing

## Directory Structure

Experiment logs are now organized within each experiment directory:
- Raw logs: `experiments/{experiment_name}/logs/`
- Dual-evaluated logs: `experiments/{experiment_name}/dual_evaluated/` 
- Human-reviewed logs: `experiments/{experiment_name}/scored_logs/`

This structure maintains complete data lineage and makes experiments self-contained for research sharing.