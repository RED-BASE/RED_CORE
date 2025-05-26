# data/

Project data, logs, and prompt artifacts.

- **logs/**: Raw and processed logs, including review sessions.
  - **review_sessions/**: Session-specific logs and summaries.
  - **raw/**: Raw log files (JSON).
- **processed/**: Cleaned and processed data files (CSV).
- **flattened/**: Flattened log files (CSV).
- **rolled/**: Placeholder for rolled-up data.
- **summary/**: Summary CSVs of logs and drift metrics.
- **personas/**: YAML files defining user/persona profiles for experiments.
- **prompts/**: Prompt templates and artifacts.
  - **system/**: System prompt YAMLs.
  - **user/**: User prompt YAMLs, organized by experiment type. 