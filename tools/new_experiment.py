import os
import sys
import shutil
from datetime import date
import argparse

TEMPLATE_PATH = "data/experiments/exp_template"
EXPERIMENTS_PATH = "data/experiments"

def main():
    parser = argparse.ArgumentParser(description="Scaffold a new experiment folder from template.")
    parser.add_argument("--name", required=True, help="Experiment name (no spaces)")
    parser.add_argument("--contributors", required=True, help="Comma-separated list")
    parser.add_argument("--purpose", required=True, help="Short description")
    args = parser.parse_args()

    exp_path = os.path.join(EXPERIMENTS_PATH, args.name)
    if os.path.exists(exp_path):
        print(f"ERROR: Experiment folder '{exp_path}' already exists.")
        sys.exit(1)

    shutil.copytree(TEMPLATE_PATH, exp_path)
    readme_path = os.path.join(exp_path, "README.md")

    with open(readme_path, "r") as f:
        content = f.read()

    content = content.replace("(Replace with the full, descriptive name of your experiment)", args.name)
    content = content.replace("YYYY-MM-DD", str(date.today()))
    content = content.replace("@github_handle, Name, etc.", args.contributors)
    content = content.replace("(State the purpose, hypothesis, or research question for this experiment)", args.purpose)

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"New experiment '{args.name}' scaffolded at {exp_path}")
    print("Next: Edit the README to complete all required sections. Commit immediately.")

if __name__ == "__main__":
    main()
