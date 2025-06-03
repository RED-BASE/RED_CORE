import json
import subprocess
import os
import argparse
import sys
import shutil
import yaml
import hashlib
from typing import Optional
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import shlex
from tqdm import tqdm
import threading
import copy
import platform

from app.core.log_schema import SessionLog, Turn
from app.core.log_utils import log_session, generate_readable_run_id
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
    get_model_snapshot_id,
)
from app.core.context import ConversationContext, ConversationHistory
from safety.containment import (
    containment_summary,
    flatten_containment_flags,
    override_output_if_flagged,
)


def compute_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

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

def compute_scenario_hash(model_name, system_prompt, user_prompt, temperature, persona=None):
    parts = [model_name, system_prompt, user_prompt, str(temperature), persona or "none"]
    return hash_string("||".join(parts))

def load_persona(persona_name: str) -> dict:
    persona_path = Path("personas") / f"{persona_name}.yaml"
    if not persona_path.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")
    return yaml.safe_load(persona_path.read_text())

def get_red_core_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "NO_GIT_HASH"

def compute_log_hash(log_dict):
    # Exclude log_hash field itself
    log_copy = copy.deepcopy(log_dict)
    log_copy.pop("log_hash", None)
    return hashlib.sha256(json.dumps(log_copy, sort_keys=True, default=str).encode("utf-8")).hexdigest()

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
    run_command: Optional[str] = None,
    model_name_pad: int = 20,
) -> dict:
    canonical_model_name = resolve_model(model_name)
    model_code = get_model_code(canonical_model_name)
    model_vendor = get_model_vendor(canonical_model_name)
    model_snapshot_id = get_model_snapshot_id(model_vendor, canonical_model_name)

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

    log_output = SessionLog(
        isbn_run_id=generate_readable_run_id(
            model_name=canonical_model_name,
            user_prompt_tag=yaml_path.stem,
            system_prompt_tag=system_prompt_tag,
            persona=persona_name,
            experiment_code=DEFAULT_EXPERIMENT_CODE,
        ),
        exploit_path=str(yaml_path),
        model=canonical_model_name,
        model_code=model_code,
        model_vendor=model_vendor,
        model_snapshot_id=model_snapshot_id,
        mode=mode,
        temperature=temperature,
        system_prompt_tag=system_prompt_tag,
        system_prompt_hash=system_prompt_hash,
        user_prompt_hash=prompt_hash,
        persona=persona_name or "none",
        turn_index_offset=1,
        experiment_id=experiment_id or yaml_path.stem,
        scenario_hash=scenario_hash or compute_scenario_hash(
            canonical_model_name,
            sys_prompt_text,
            raw_yaml_text,
            temperature,
            persona_name
        ),
        turns=[],
        evaluator_version="unknown",
        run_command=run_command,
        sdk_version=openai.__version__,
        python_version=sys.version,
        red_core_version=get_red_core_version(),
        created_at=datetime.now().isoformat(),
        status="complete",
        tags=[],
        provenance=[],
        runtime=platform.platform(),
    )

    variants = exploit_data.get("variants", [])
    total_turns = len(variants)
    model_label = (canonical_model_name + ' ' + '.' * model_name_pad)[:model_name_pad]
    print(f"{model_label} 0/{total_turns} turns complete", end="\r", flush=True)
    turns_complete = 0
    for variant in variants:
        raw = variant.get("prompt", "")
        if not raw.strip():
            print(f"⚠️ Skipping blank variant: {variant.get('id', '[no id]')}")
            turns_complete += 1
            print(f"{model_label} {turns_complete}/{total_turns} turns complete", end="\r", flush=True)
            continue
        if "\n" in raw:
            header, body = raw.split("\n", 1)
        else:
            header, body = "", raw
        prompt_body = body.strip()
        turn_index = len(log_output.turns) + log_output.turn_index_offset
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
        result = runner.generate(prompt_body, **generate_kwargs)
        ctx.update_output(result["model_output"])
        ctx.meta.update({
            "vendor_model_id": result.get("model_name"),
            "usage": result.get("usage"),
        })
        history.append_assistant(ctx.model_output)
        summary = containment_summary(ctx.user_input, ctx.rendered_prompt, ctx.model_output)
        flags = flatten_containment_flags(summary)
        if flags and not disable_containment:
            ctx.update_output(override_output_if_flagged(ctx.model_output, flags))
        turn_obj = Turn(
            review_status="pending",
            timestamp=None,
            turn_index=turn_index,
            rendered_prompt=prompt_body,
            user_input=ctx.user_input,
            model_output=ctx.model_output,
            persona=ctx.persona,
            variant_id=variant.get("id"),
            containment_flags=flags,
            containment_summary=summary,
            drift_score=None,
            refusal_score=None,
            notes=None,
            reviewer=None,
            tags=[],
        )
        log_output.turns.append(turn_obj)
        turns_complete += 1
        print(f"{model_label} {turns_complete}/{total_turns} turns complete", end="\r", flush=True)
    print(f"{model_label} {turns_complete}/{total_turns} turns complete")

    # Compute log_hash before saving
    log_dict = log_output.model_dump()
    log_hash = compute_log_hash(log_dict)
    log_output.log_hash = log_hash
    log_path = Path(LOG_DIR) / f"{log_output.isbn_run_id}.json"
    log_session(str(log_path), log_output)
    print(f"📁 Log saved to: {log_path}")
    return log_output

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser("new", help="Scaffold a new experiment folder from template.")
    new_parser.add_argument("--name", required=True)
    new_parser.add_argument("--contributors", required=True)
    new_parser.add_argument("--purpose", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiment(s) with specified models and prompts.")
    run_parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude-3-opus", "gemini-pro"])
    run_parser.add_argument("--sys-prompt", required=True)
    run_parser.add_argument("--usr-prompt", required=True)
    run_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_parser.add_argument("--mode", choices=["audit", "attack"], default=DEFAULT_MODE)
    run_parser.add_argument("--persona")
    run_parser.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    run_parser.add_argument("--disable-containment", action="store_true")
    run_parser.add_argument("--experiment-id")
    run_parser.add_argument("--scenario-hash")
    run_parser.add_argument("--score-log", action="store_true")

    args, _ = parser.parse_known_args()

    if args.command == "new":
        return

    elif args.command == "run" or args.command is None:
        sys_prompt_path = Path(args.sys_prompt)
        usr_prompt_path = Path(args.usr_prompt)
        print(f"\n📂 System prompt: {sys_prompt_path}")
        print(f"📂 User prompt:   {usr_prompt_path}")
        print(f"🧪 Models:        {args.models}")
        if args.disable_containment:
            print("⚠️  Containment disabled")
        print("")

        # Capture the command used to run the script
        run_command_str = "PYTHONPATH=. " + " ".join([shlex.quote(arg) for arg in sys.argv])

        # Calculate max model name length for padding
        model_name_pad = max(len(m) for m in args.models) + 8
        successes = []
        failures = []
        lock = threading.Lock()
        log_dir_path = Path(LOG_DIR)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        def run_one(model):
            try:
                result = run_exploit_yaml(
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
                    run_command=run_command_str,
                    model_name_pad=model_name_pad,
                )
                with lock:
                    successes.append(model)
                return result
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                with lock:
                    failures.append((model, str(e), tb))
                return None
        # Simple model run counter
        total_models = len(args.models)
        models_complete = 0
        print(f"Model runs: 0/{total_models} complete")
        with ThreadPoolExecutor(max_workers=total_models) as executor:
            futures = {executor.submit(run_one, m): m for m in args.models}
            for future in as_completed(futures):
                _ = future.result()
                models_complete += 1
                print(f"Model runs: {models_complete}/{total_models} complete")

        # Print summary
        print("")
        print(f"✅ {len(successes)} logs saved to: {log_dir_path}")
        if failures:
            print(f"❌ {len(failures)} model(s) failed:")
            for model, err, _ in failures:
                print(f"    - {model}: {err}")
            # Write all errors to a run_failures.txt file
            error_log_path = log_dir_path / "run_failures.txt"
            with open(error_log_path, "w") as f:
                for model, err, tb in failures:
                    f.write(f"Model: {model}\nError: {err}\nTraceback:\n{tb}\n{'-'*40}\n")
            print(f"  (See {error_log_path} for details)")
        else:
            print("All models completed successfully!")

if __name__ == "__main__":
    main()
