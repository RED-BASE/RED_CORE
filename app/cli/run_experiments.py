import subprocess
import os
import argparse
import sys
from pathlib import Path
import json
import yaml
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.hash_utils import hash_string
from app.config import (
    LOG_DIR,
    DEFAULT_TEMPERATURE,
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_MODE,
    DEFAULT_EXPERIMENT_CODE,
    resolve_model,
    get_model_code,
    get_model_vendor,
)
from app.core.context import ConversationContext, ConversationHistory
from safety.containment import containment_summary, flatten_containment_flags, override_output_if_flagged
from app.core.log_utils import generate_readable_run_id
import shutil
from datetime import date

# --- Runner Factory Logic ---
def get_runner(model_name: str):
    canonical_name = resolve_model(model_name)
    vendor = get_model_vendor(canonical_name)
    if vendor == "openai":
        from app.api_runners.openai_runner import OpenAIRunner
        return OpenAIRunner(model_name=canonical_name)
    elif vendor == "anthropic":
        from app.api_runners.anthropic_runner import AnthropicRunner
        return AnthropicRunner(model_name=canonical_name)
    elif vendor == "google":
        from app.api_runners.google_runner import GoogleRunner
        return GoogleRunner(model_name=canonical_name)
    elif vendor == "local":
        from app.api_runners.llama_cpp_runner import LlamaCppRunner
        return LlamaCppRunner(model_path=canonical_name)
    else:
        raise ValueError(f"No runner registered for model '{model_name}' (resolved as '{canonical_name}')")

# --- Helper Functions ---
def compute_scenario_hash(model_name, system_prompt, user_prompt, temperature, persona=None):
    parts = [model_name, system_prompt, user_prompt, str(temperature), persona or "none"]
    return hash_string("||".join(parts))

def load_persona(persona_name: str) -> dict:
    persona_path = Path("personas") / f"{persona_name}.yaml"
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")
    return yaml.safe_load(persona_path.read_text())

# --- Main Experiment Logic ---
def run_exploit_yaml(
    yaml_path: str,
    sys_prompt: str,
    model_name: str = "gpt-4",
    temperature: float = DEFAULT_TEMPERATURE,
    mode: str = DEFAULT_MODE,
    persona_name: Optional[str] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    disable_containment: bool = False,
    experiment_id: Optional[str] = None,
    scenario_hash: Optional[str] = None,
    score_log: bool = False,
) -> dict:
    canonical_model_name = resolve_model(model_name)
    model_code = get_model_code(canonical_model_name)
    model_vendor = get_model_vendor(canonical_model_name)
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)
    raw_yaml_text = yaml_path.read_text()
    exploit_data = yaml.safe_load(raw_yaml_text)
    prompt_hash = hash_string(raw_yaml_text)
    sys_prompt_path = Path(sys_prompt)
    if not sys_prompt_path.exists():
        raise FileNotFoundError(sys_prompt_path)
    sys_prompt_text = sys_prompt_path.read_text()
    system_prompt_hash = hash_string(sys_prompt_text)
    system_prompt_tag = sys_prompt_path.stem + ":latest"
    runner = get_runner(canonical_model_name)
    if hasattr(runner, "set_system_prompt"):
        runner.set_system_prompt(sys_prompt_text)
    if persona_name:
        persona_blob = load_persona(persona_name)
        if hasattr(runner, "set_persona"):
            runner.set_persona(persona_blob)
    history = ConversationHistory(system_prompt=sys_prompt_text)
    log_output = {
        "isbn_run_id": generate_readable_run_id(
            model_name=canonical_model_name,
            user_prompt_tag=yaml_path.stem,
            system_prompt_tag=system_prompt_tag,
            persona=persona_name,
            experiment_code=DEFAULT_EXPERIMENT_CODE,
        ),
        "exploit_path": str(yaml_path),
        "model": canonical_model_name,
        "model_code": model_code,
        "model_vendor": model_vendor,
        "mode": mode,
        "temperature": temperature,
        "system_prompt_tag": system_prompt_tag,
        "system_prompt_hash": system_prompt_hash,
        "user_prompt_hash": prompt_hash,
        "persona": persona_name or "none",
        "turns": [],
        "turn_index_offset": 1,
        "experiment_id": experiment_id or yaml_path.stem,
        "scenario_hash": scenario_hash or compute_scenario_hash(
            canonical_model_name,
            sys_prompt_text,
            raw_yaml_text,
            temperature,
            persona_name
        ),
    }
    for variant in exploit_data.get("variants", []):
        raw = variant.get("prompt", "")
        if not raw.strip():
            print(f"⚠️ Skipping blank variant: {variant.get('id', '[no id]')}")
            continue
        if "\n" in raw:
            header, body = raw.split("\n", 1)
        else:
            header, body = "", raw
        prompt_body = body.strip()
        turn_index = len(log_output["turns"]) + 1
        ctx = ConversationContext(
            rendered_prompt=prompt_body,
            persona=persona_name or "none",
            system_prompt_tag=system_prompt_tag,
            meta={
                "variant_id": variant.get("id"),
                "prompt_header": header.strip(),
            },
        )
        ctx.user_input = prompt_body
        history.append_user(prompt_body)
        generate_kwargs = {
            "temperature": temperature,
            "turn_index": turn_index,
        }
        if model_vendor == "google":
            generate_kwargs["conversation"] = history
        ctx.update_output(
            runner.generate(
                prompt_body,
                **generate_kwargs
            )
        )
        history.append_assistant(ctx.model_output)
        summary = containment_summary(
            ctx.user_input, ctx.rendered_prompt, ctx.model_output
        )
        flags = flatten_containment_flags(summary)
        if flags and not disable_containment:
            ctx.update_output(override_output_if_flagged(ctx.model_output, flags))
        turn = {
            "turn_index": len(log_output["turns"]) + log_output["turn_index_offset"],
            **ctx.as_dict(),
            "system_prompt_text": sys_prompt_text,
            "containment_flags": flags,
            "containment_summary": summary,
            "drift_score": None,
            "drift_notes": None,
            "review_status": "pending"
        }
        log_output["turns"].append(turn)
    # File writing and scoring handled here
    logs_dir = Path(LOG_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_path = logs_dir / f"{log_output['isbn_run_id']}.json"
    save_path.write_text(json.dumps(log_output, indent=2))
    print(f"📁 Log saved to: {save_path}")

def create_experiment_folder(name: str, contributors: str, purpose: str):
    TEMPLATE_PATH = "data/experiments/exp_template"
    EXPERIMENTS_PATH = "data/experiments"
    exp_path = os.path.join(EXPERIMENTS_PATH, name)
    if os.path.exists(exp_path):
        print(f"ERROR: Experiment folder '{exp_path}' already exists.")
        sys.exit(1)
    shutil.copytree(TEMPLATE_PATH, exp_path)
    readme_path = os.path.join(exp_path, "README.md")
    with open(readme_path, "r") as f:
        content = f.read()
    content = content.replace("(Replace with the full, descriptive name of your experiment)", name)
    content = content.replace("YYYY-MM-DD", str(date.today()))
    content = content.replace("@github_handle, Name, etc.", contributors)
    content = content.replace("(State the purpose, hypothesis, or research question for this experiment)", purpose)
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"New experiment '{name}' scaffolded at {exp_path}")
    print("Next: Edit the README to complete all required sections. Commit immediately.")
    return {"exp_path": exp_path, "readme_content": content}

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: new experiment
    new_parser = subparsers.add_parser("new", help="Scaffold a new experiment folder from template.")
    new_parser.add_argument("--name", required=True, help="Experiment name (no spaces)")
    new_parser.add_argument("--contributors", required=True, help="Comma-separated list")
    new_parser.add_argument("--purpose", required=True, help="Short description")

    # Subcommand: run (default)
    run_parser = subparsers.add_parser("run", help="Run experiment(s) with specified models and prompts.")
    run_parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude-3-opus", "gemini-pro"], help="List of models to run in parallel")
    run_parser.add_argument("--sys-prompt", required=True, help="Path to system prompt file")
    run_parser.add_argument("--usr-prompt", required=True, help="Path to user prompt YAML")
    run_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_parser.add_argument("--mode", choices=["audit", "attack"], default=DEFAULT_MODE)
    run_parser.add_argument("--persona", help="Optional persona name (file in personas/)")
    run_parser.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    run_parser.add_argument("--disable-containment", action="store_true")
    run_parser.add_argument("--experiment-id", type=str, default=None)
    run_parser.add_argument("--scenario-hash", type=str, default=None)
    run_parser.add_argument("--score-log", action="store_true", help="Run drift scoring after saving the log")

    args, extra_args = parser.parse_known_args()

    if args.command == "new":
        create_experiment_folder(args.name, args.contributors, args.purpose)
        return
    elif args.command == "run" or args.command is None:
        # Existing run logic
        sys_prompt_path = Path(args.sys_prompt)
        usr_prompt_path = Path(args.usr_prompt)
        print(f"\n📂 System prompt: {sys_prompt_path}")
        print(f"📂 User prompt:   {usr_prompt_path}")
        print(f"🧪 Models:        {args.models}")
        if args.disable_containment:
            print("⚠️  Containment disabled")
        print("")
        def run_one(model):
            return run_exploit_yaml(
                yaml_path=str(usr_prompt_path),
                sys_prompt=str(sys_prompt_path),
                model_name=model,
                temperature=args.temperature,
                mode=args.mode,
                persona_name=args.persona,
                drift_threshold=args.drift_threshold,
                disable_containment=args.disable_containment,
                experiment_id=args.experiment_id,
                scenario_hash=args.scenario_hash,
                score_log=args.score_log,
            )
        with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
            futures = {executor.submit(run_one, m): m for m in args.models}
            for future in as_completed(futures):
                result = future.result()
                print(f"✅ {futures[future]} finished successfully")

if __name__ == "__main__":
    main()
