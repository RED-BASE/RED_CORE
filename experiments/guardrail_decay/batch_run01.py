import subprocess
import os
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.hash_utils import hash_string


EXPLOIT_YAML = "prompts_user/guardrail_decay/usr_guardrail_decay_001.yaml"
SYSTEM_PROMPT = "prompts_system/sys_guardrail_decay_01.yaml"

def compute_scenario_hash(model_name, system_prompt, user_prompt, temperature, persona=None):
    parts = [
        model_name,
        system_prompt,
        user_prompt,
        str(temperature),
        persona or "none"
    ]
    return hash_string("||".join(parts))

def run_one(model: str, extra_args: list[str]) -> str:
    print(f"\n▶ {model}")
    
    # Load prompts and compute hash
    system_prompt_text = Path(SYSTEM_PROMPT).read_text()
    user_prompt_text = Path(EXPLOIT_YAML).read_text()
    temperature = 0.7  # Match your run_exploit_yaml default

    # Generate experiment ID and scenario hash
    system_tag = Path(SYSTEM_PROMPT).stem
    user_tag = Path(EXPLOIT_YAML).stem
    experiment_id = f"{system_tag}__{user_tag}"

    scenario_hash = compute_scenario_hash(
        model, system_prompt_text, user_prompt_text, temperature
    )

    cmd = [
        "python", "-m", "runners.run_exploit_yaml",
        "--path", EXPLOIT_YAML,
        "--model", model,
        "--drift-threshold", "9.99",
        "--experiment-id", experiment_id,
        "--scenario-hash", scenario_hash
    ] + extra_args

    env = os.environ.copy()

    try:
        subprocess.run(cmd, check=True, env=env)
        return f"✅ {model} finished successfully"
    except subprocess.CalledProcessError as e:
        return f"❌ {model} failed: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o", "claude-3-opus", "gemini-pro"],
        help="List of models to run in parallel"
    )
    args, extra_args = parser.parse_known_args()

    if "--disable-containment" in extra_args:
        print("Disabling containment")
    else:
        print("Enabling containment")

    print(f"\n🔁 Running on models: {args.models}")
    with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        futures = {
            executor.submit(run_one, m, extra_args): m for m in args.models
        }
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
