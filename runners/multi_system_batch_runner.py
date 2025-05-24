import argparse
import concurrent.futures
from pathlib import Path
import hashlib

from runners.run_exploit_yaml import run_exploit_yaml

# --- CONFIG IMPORT ---
try:
    from config import MODEL_ALIASES, DEFAULT_TEMPERATURE
except ImportError:
    # Safe fallback if config.py is missing, but warn the user
    MODEL_ALIASES = {}
    DEFAULT_TEMPERATURE = 0.7

def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def compute_scenario_hash(sys_prompt_text, usr_prompt_text, model, experiment_id):
    key = f"{sys_prompt_text}\n{usr_prompt_text}\n{model}\n{experiment_id}"
    return hash_string(key)[:6].upper()

def extract_prompt_tag(prompt_path):
    return Path(prompt_path).stem

def parse_extra_args(extra):
    out = {}
    for i in range(0, len(extra or []), 2):
        key = extra[i].lstrip("-").replace("-", "_")
        val = extra[i + 1]
        if isinstance(val, str):
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.isdigit():
                val = int(val)
            else:
                try:
                    val = float(val)
                except ValueError:
                    pass
        out[key] = val
    return out

def resolve_model(model):
    """Return canonical model name using aliases in config, if present."""
    return MODEL_ALIASES.get(model.lower(), model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sys-prompts", nargs="+", required=True)
    parser.add_argument("--usr-prompt", required=True)
    parser.add_argument("--score-log", action="store_true")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print("🧪 Models:", ", ".join(args.models))
    print(f"🧪 Reps: {args.reps}")
    print("🧪 Permutations:")
    for sys_prompt_path in args.sys_prompts:
        sys_tag = extract_prompt_tag(sys_prompt_path)
        usr_tag = extract_prompt_tag(args.usr_prompt)
        print(f"  - {sys_tag} × {usr_tag}")
    print()

    extra_kwargs = parse_extra_args(args.extra_args)
    total_runs = len(args.models) * len(args.sys_prompts) * args.reps
    completed = 0
    futures = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for rep in range(args.reps):
            for sys_prompt_path in args.sys_prompts:
                for model in args.models:
                    sys_prompt_text = Path(sys_prompt_path).read_text()
                    usr_prompt_text = Path(args.usr_prompt).read_text()
                    canonical_model = resolve_model(model)

                    scenario_hash = compute_scenario_hash(
                        sys_prompt_text=sys_prompt_text,
                        usr_prompt_text=usr_prompt_text,
                        model=canonical_model,
                        experiment_id=args.experiment_id
                    )

                    sys_tag = extract_prompt_tag(sys_prompt_path)
                    usr_tag = extract_prompt_tag(args.usr_prompt)
                    dynamic_experiment_id = f"{args.experiment_id}__{sys_tag}__{usr_tag}__{canonical_model}__r{rep+1}"

                    futures.append(executor.submit(
                        run_exploit_yaml,
                        model_name=canonical_model,
                        sys_prompt=sys_prompt_path,
                        yaml_path=args.usr_prompt,
                        experiment_id=dynamic_experiment_id,
                        score_log=args.score_log,
                        **extra_kwargs
                    ))

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            print(f"Progress: [{completed:>3}/{total_runs}]  ({result})", end="\r")

    print(f"\n✅ All {total_runs} runs complete. 😎")

if __name__ == "__main__":
    main()
