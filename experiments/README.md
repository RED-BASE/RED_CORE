# experiments/

ALWAYS START A NEW EXP FOLDER BY COPYING THE TEMPLATE AND UPDATING THE README
NO EXCUSES

Experiment-specific code and results.
- **exp_template/**: Copy this template and update the readme whenever you do. 
- **guardrail_decay/**: Guardrail decay experiment scripts, logs, and documentation.

## Log Workflow

- Raw logs are saved to `logs/` inside each experiment folder.
- Scored logs are uploaded to `scored_logs/` via the iOS scoring app and merged into the repo via pull request.
- Each log includes robust error handling, per-turn token counts, latency, and a unified summary output at the end of each run.
- The review process is manual (iOS or desktop), and all scored logs are merged via PR for full auditability.