import subprocess
import os
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.hash_utils import hash_string


def compute_scenario_hash(model_name, system_prompt, user_prompt, temperature, persona=None):
    parts = [
        model_name,
        system_prompt,
        user_prompt,
        str(temperature),
        persona or "none"
    ]
    return hash_string("||".join(parts))


def run_one(model: str, sys_prompt_path: Path, usr_prompt_path: Path, extra_args: list[str]) -> str:
    print(f"\n▶ {model}")

    # Load prompt content and compute hash
    system_prompt_text = sys_prompt_path.read_text()
    user_prompt_text = usr_prompt_path.read_text()
    temperature = 0.7

    system_tag = sys_prompt_path.stem
    user_tag = usr_prompt_path.stem
    experiment_id = f"{system_tag}__{user_tag}"

    scenario_hash = compute_scenario_hash(
        model, system_prompt_text, user_prompt_text, temperature
    )

    cmd = [
        "python", "-m", "runners.run_exploit_yaml",
        "--path", str(usr_prompt_path),
        "--model", model,
        "--drift-threshold", "9.99",
        "--experiment-id", experiment_id,
        "--scenario-hash", scenario_hash,
        "--prompt", f"{system_tag}:latest"
    ] + extra_args

    env = os.environ.copy()

    try:
        subprocess.run(cmd, check=True, env=env)
        return f"✅ {model} finished successfully"
    except subprocess.CalledProcessError as e:
        return f"❌ {model} failed: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude-3-opus", "gemini-pro"],
                        help="List of models to run in parallel")
    parser.add_argument("--sys-prompt", required=True, help="Path to system prompt file")
    parser.add_argument("--usr-prompt", required=True, help="Path to user prompt YAML")
    args, extra_args = parser.parse_known_args()

    sys_prompt_path = Path(args.sys_prompt)
    usr_prompt_path = Path(args.usr_prompt)

    print(f"\n📂 System prompt: {sys_prompt_path}")
    print(f"📂 User prompt:   {usr_prompt_path}")
    print(f"🧪 Models:        {args.models}")
    if "--disable-containment" in extra_args:
        print("⚠️  Containment disabled")
    print("")

    with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        futures = {
            executor.submit(run_one, m, sys_prompt_path, usr_prompt_path, extra_args): m
            for m in args.models
        }
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
