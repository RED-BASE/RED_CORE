README Addendum—Enforcement Rules
🚫 Dependency Management
ALWAYS use:


poetry add <package>
NEVER use:


pip install <anything>
Violation = rejected commit.

🔧 Installation
This project is script-driven, not a Python package.

Install dependencies with:


poetry install --no-root
All automation and tools must be run via:


poetry run <tool or script>

🚀 Running Experiments
All experiment runs are orchestrated through a single entrypoint:

poetry run run-experiments --models <model1> <model2> ... --sys-prompt <system_prompt_file> --usr-prompt <user_prompt_yaml> [other options]

- All runner instantiation, prompt handling, and logging is managed by this script.
- No other script or subprocess should be used to run experiments.
- See --help for all available arguments (temperature, persona, drift-threshold, etc).

🛡️ File I/O Enforcement
A pre-commit hook enforces strict provenance and reproducibility:
- **All file write/mutation operations** (e.g., open with 'w', 'a', or 'x'; .to_csv; .to_json; .write_json; os.rename; os.remove; shutil.move; shutil.copy) are **blocked** in all scripts **except**:
  - `app/cli/run_experiments.py` (orchestrator and experiment creation)
  - `meta/friction_audit.py` (meta-tool)
  - `dev_log/reflections/create_log_entry.sh` (dev log)
- **Read operations** are always allowed for config, models, or data.
- If a forbidden write is detected, the commit is blocked with:
  > "Commit blocked: File write operation detected in [path]. Only allowed in orchestrator and meta-tools. See CONTRIBUTING.md for allowed patterns."

**Rationale:**
- Writing is where provenance and reproducibility break down. All mutations must be orchestrated and auditable.

📝 dev_log/
dev_log/ is strictly for developer reflection and process notes. The only allowed script is dev_log/reflections/create_log_entry.sh for structured log entry creation. No code, data, or other scripts—ever.

🧱 Experiment Folders
All new experiments MUST be created with:

poetry run run-experiments new --name <experiment_name> --contributors <contributors> --purpose <purpose>

Never copy folders by hand. Never skip README auto-fill. All file creation is enforced and tracked.# RED_CORE
