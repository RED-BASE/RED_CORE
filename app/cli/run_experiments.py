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
import time
import itertools
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text
import sys
import termios
import tty

from app.core.log_schema import SessionLog, Turn, ReproductionInfo, EvaluatorVersions
from app.core.log_utils import log_session, generate_readable_run_id
from app.core.hash_utils import hash_string
from app.core.logger import get_experiment_logger
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

def get_available_experiments():
    """Get list of available experiment folders."""
    experiment_dirs = []
    if Path("experiments").exists():
        for path in Path("experiments").iterdir():
            if path.is_dir() and not path.name.startswith("."):
                experiment_dirs.append(path.name)
    return sorted(experiment_dirs)

def get_available_prompts(pattern, base_path="data/prompts"):
    """Get available prompt files matching pattern."""
    if Path(base_path).exists():
        files = list(Path(base_path).rglob(pattern))
        return sorted([str(f) for f in files])  # Keep full path, sorted
    return []

def get_available_personas():
    """Get available persona files."""
    if Path("personas").exists():
        personas = [f.stem for f in Path("personas").glob("*.yaml")]
        return ["none"] + sorted(personas)
    return ["none"]

def select_from_list(console, title, options, allow_multiple=False):
    """Interactive selection with arrow keys using Rich's built-in prompt."""
    from rich.prompt import Prompt
    
    if not options:
        console.print(f"[red]No {title.lower()} available[/red]")
        return None
    
    # Show the options
    console.print(f"[bold cyan]{title}:[/bold cyan]")
    for i, option in enumerate(options):
        console.print(f"  {i+1}. {option}")
    
    if allow_multiple:
        console.print("[dim]Enter numbers separated by spaces (e.g., 1 3 5)[/dim]")
        while True:
            choice = Prompt.ask("Select options")
            try:
                indices = [int(x.strip()) - 1 for x in choice.split()]
                if all(0 <= i < len(options) for i in indices):
                    return [options[i] for i in indices]
                else:
                    console.print("[red]Invalid selection. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter numbers only.[/red]")
    else:
        while True:
            choice = Prompt.ask("Select option (number)")
            try:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                else:
                    console.print(f"[red]Please enter a number between 1 and {len(options)}[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

def prompt_for_text(console, prompt, required=True):
    """Simple text input prompt."""
    while True:
        value = Prompt.ask(prompt)
        if not required or value.strip():
            return value.strip()
        console.print("[red]This field is required[/red]")

def configure_experiment_interactively():
    """Interactive experiment configuration using Rich with arrow key navigation."""
    console = Console()
    
    console.print()
    console.print(Panel("⚙ Interactive Batch Configuration", style="bold blue"))
    console.print()
    
    # Get experiment folder
    experiments = get_available_experiments()
    experiment_choices = ["none"] + experiments
    experiment = select_from_list(console, "Select Experiment Folder", experiment_choices)
    if experiment is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return None
    if experiment == "none":
        experiment = None
    
    console.print()
    
    # System prompts
    sys_prompts_options = get_available_prompts("**/sys*.yaml")
    if not sys_prompts_options:
        console.print("[red]No system prompts found![/red]")
        return None
    
    sys_prompts = select_from_list(console, "Select System Prompts", sys_prompts_options, allow_multiple=True)
    if not sys_prompts:
        console.print("[yellow]No system prompts selected.[/yellow]")
        return None
    
    console.print()
    
    # User prompts - search in all subfolders
    usr_prompts_options = get_available_prompts("**/*.yaml", "data/prompts/user")
    if experiment:
        # Look for experiment-specific prompts first
        exp_prompts = get_available_prompts(f"**/*{experiment}*.yaml", "data/prompts/user")
        if exp_prompts:
            console.print(f"[green]Found {len(exp_prompts)} experiment-specific prompts[/green]")
            usr_prompts_options = exp_prompts + usr_prompts_options
    
    if not usr_prompts_options:
        console.print("[red]No user prompts found![/red]")
        return None
        
    usr_prompts = select_from_list(console, "Select User Prompts", usr_prompts_options, allow_multiple=True)
    if not usr_prompts:
        console.print("[yellow]No user prompts selected.[/yellow]")
        return None
    
    console.print()
    
    # Models
    model_choices = ["gpt-4o", "claude-3-7-sonnet-20250219", "claude-3-opus", "gemini-pro", "gemini-1.5-pro"]
    models = select_from_list(console, "Select Models", model_choices, allow_multiple=True)
    if not models:
        console.print("[yellow]No models selected.[/yellow]")
        return None
    
    console.print()
    
    # Personas
    personas_options = get_available_personas()
    personas = select_from_list(console, "Select Personas", personas_options, allow_multiple=True)
    if not personas:
        personas = ["none"]  # Default to none if nothing selected
    
    console.print()
    
    # Parameters
    repetitions = int(prompt_for_text(console, "Repetitions per combination"))
    temperature = float(prompt_for_text(console, "Temperature"))
    experiment_code = prompt_for_text(console, "Experiment code (e.g., 80K, GRD)")
    disable_containment = Confirm.ask("Disable containment?")
    
    # Calculate totals
    total_combinations = len(sys_prompts) * len(usr_prompts) * len(models) * len(personas)
    total_runs = total_combinations * repetitions
    
    # Show summary
    console.print()
    summary_table = Table(title="Configuration Summary", style="cyan")
    summary_table.add_column("Setting", style="white")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("System prompts", f"{len(sys_prompts)} files")
    summary_table.add_row("User prompts", f"{len(usr_prompts)} files")
    summary_table.add_row("Models", f"{len(models)} models")
    summary_table.add_row("Personas", f"{len(personas)} variations")
    summary_table.add_row("Repetitions", str(repetitions))
    summary_table.add_row("[bold]Total runs[/bold]", f"[bold]{total_runs}[/bold] ({len(sys_prompts)}×{len(usr_prompts)}×{len(models)}×{len(personas)}×{repetitions})")
    
    console.print(summary_table)
    console.print()
    
    if not Confirm.ask("Proceed with this configuration?"):
        console.print("[yellow]Aborted.[/yellow]")
        return None
    
    return {
        'sys_prompts': sys_prompts,
        'usr_prompts': usr_prompts,
        'models': models,
        'personas': personas,
        'repetitions': repetitions,
        'temperature': temperature,
        'experiment_code': experiment_code,
        'disable_containment': disable_containment,
        'experiment': experiment
    }

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
    user_turn_callback=None,
    experiment_code: str = DEFAULT_EXPERIMENT_CODE,
    quiet: bool = False,
    log_dir: Optional[str] = None,
) -> dict:
    """Run an exploit YAML file and return the generated log.

    Args:
        yaml_path: Path to the YAML file containing user prompt variants.
        sys_prompt: Path to the system prompt text that will be sent to the model.
        model_name: Canonical model name or alias to run the prompts against.
        temperature: Sampling temperature for generation.
        mode: Execution mode such as ``audit`` or ``attack``.
        persona_name: Optional persona configuration to load.
        drift_threshold: Drift threshold used when scoring logs.
        disable_containment: If ``True`` containment overrides are skipped.
        experiment_id: Optional identifier recorded in the resulting log.
        scenario_hash: Optional precomputed scenario hash for reproducibility.
        score_log: Whether to compute drift/refusal metrics for the run.
        run_command: Command string recorded for provenance.
        model_name_pad: Padding width used when printing model progress.
        user_turn_callback: Optional callback invoked after each user turn.
        experiment_code: Experiment code used in log naming and metadata.

    Returns:
        A :class:`SessionLog` object describing the completed run.
    """
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
            experiment_code=experiment_code,
        ),
        exploit_path=str(yaml_path),
        model=canonical_model_name,
        model_vendor=model_vendor,
        mode=mode,
        temperature=temperature,
        
        # Enhanced prompt content and references
        system_prompt_tag=system_prompt_tag,
        system_prompt_content=sys_prompt_text,  # NEW: Full system prompt text
        system_prompt_file=str(sys_prompt_path.relative_to(Path.cwd())),  # NEW: Relative file path
        system_prompt_hash=system_prompt_hash,
        user_prompt_file=str(yaml_path.relative_to(Path.cwd())),  # NEW: Relative file path
        user_prompt_hash=prompt_hash,
        
        # Experimental context (basic detection)
        experiment_readme=f"experiments/{experiment_code.lower()}/README.md" if experiment_code else None,  # NEW
        hypothesis=exploit_data.get("hypothesis"),  # NEW: Extract from YAML if present
        
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
        
        # Enhanced provenance tracking
        reproduction_info=ReproductionInfo(  # NEW: Structured reproduction data
            experiment_code=experiment_code,
            model=canonical_model_name,
            system_prompt=system_prompt_tag,
            user_prompt=yaml_path.stem,
            persona=persona_name,
            temperature=temperature
        ),
        evaluator_versions=EvaluatorVersions(  # NEW: Tool version tracking
            red_core_schema="2.0.0",  # Schema version
            llm_evaluator=None  # Will be populated when LLM evaluation runs
        ),
        red_core_version=get_red_core_version(),
        
        # Legacy fields (for compatibility)
        evaluator_version="unknown",
        run_command=run_command,
        sdk_version=openai.__version__,
        python_version=sys.version,
        created_at=datetime.now().isoformat(),
        status="complete",
        tags=[],
        provenance=[],
        runtime=platform.platform(),
    )

    variants = exploit_data.get("variants", [])
    total_turns = len(variants)
    model_label = (canonical_model_name + ' ' + '.' * model_name_pad)[:model_name_pad]
    for i, variant in enumerate(variants):
        raw = variant.get("prompt", "")
        if not raw.strip():
            print(f"[WARNING] Skipping blank variant: {variant.get('id', '[no id]')}")
            continue
        if "\n" in raw:
            header, body = raw.split("\n", 1)
        else:
            header, body = "", raw
        prompt_body = body.strip()
        # Remove unnecessary quotes from YAML literal blocks  
        if prompt_body.startswith('"') and prompt_body.endswith('"'):
            prompt_body = prompt_body[1:-1]
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
        if model_vendor in {"google", "openai"}:
            generate_kwargs["conversation"] = history
        start_time = time.time()
        result = runner.generate(prompt_body, **generate_kwargs)
        latency_ms = (time.time() - start_time) * 1000
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
            turn_index=turn_index,
            user_input_id=variant.get("id"),
            persona=ctx.persona,
            raw_user_input=prompt_body,
            rendered_user_input=ctx.rendered_prompt,
            model_output=ctx.model_output,
            latency_ms=latency_ms,
            containment_flags=flags,
            containment_summary=summary,
            review_status="pending",
            drift_score=None,
            refusal_score=None,
            notes=None,
            reviewer=None,
            tags=[],
            input_token_count=result.get('usage', {}).get('prompt_tokens') if result.get('usage') else None,
            output_token_count=result.get('usage', {}).get('completion_tokens') if result.get('usage') else None,
            total_token_count=result.get('usage', {}).get('total_tokens') if result.get('usage') else None,
        )
        log_output.turns.append(turn_obj)
        if user_turn_callback:
            user_turn_callback(canonical_model_name, i+1, ctx.model_output)

    # Compute log_hash before saving
    log_dict = log_output.model_dump()
    log_hash = compute_log_hash(log_dict)
    log_output.log_hash = log_hash
    # Use provided log_dir or fall back to global LOG_DIR
    target_log_dir = log_dir if log_dir else LOG_DIR
    log_path = Path(target_log_dir) / f"{log_output.isbn_run_id}.json"
    log_session(str(log_path), log_output)
    if not quiet:
        print(f"[INFO] Log saved to: {log_path}")
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
    run_parser.add_argument("--sys-prompt")
    run_parser.add_argument("--usr-prompt")
    run_parser.add_argument("--interactive", action="store_true", help="Use interactive configuration mode")
    run_parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per experiment combination")
    run_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_parser.add_argument("--mode", choices=["audit", "attack"], default=DEFAULT_MODE)
    run_parser.add_argument("--persona")
    run_parser.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    run_parser.add_argument("--disable-containment", action="store_true")
    run_parser.add_argument("--experiment-id")
    run_parser.add_argument("--scenario-hash")
    run_parser.add_argument("--score-log", action="store_true")
    run_parser.add_argument("--experiment-code", default=DEFAULT_EXPERIMENT_CODE)

    args, _ = parser.parse_known_args()

    if args.command == "new":
        return

    elif args.command == "run" or args.command is None:
        # Check if we need interactive mode
        if args.interactive or not args.sys_prompt or not args.usr_prompt:
            config = configure_experiment_interactively()
            if not config:
                return
            
            # Override args with interactive config
            args.models = config['models']
            args.temperature = config['temperature']
            args.repetitions = config['repetitions']
            args.disable_containment = config['disable_containment']
            args.experiment_code = config['experiment_code']
            
            # Use all prompts from interactive config
            sys_prompts = config['sys_prompts']
            usr_prompts = config['usr_prompts']
        else:
            # Command line mode - single prompts
            sys_prompts = [args.sys_prompt]
            usr_prompts = [args.usr_prompt]
            
        print("")
        print("⚙ BATCH CONFIG")
        print("")
        print(f"· System prompts: {len(sys_prompts)} files")
        print(f"· User prompts: {len(usr_prompts)} files")
        print(f"· Models: {args.models}")
        if args.disable_containment:
            print("· Warning: Containment disabled")
        print("")

        # Capture the command used to run the script
        run_command_str = "PYTHONPATH=. " + " ".join([shlex.quote(arg) for arg in sys.argv])

        # Calculate total experiments and set log directory
        personas = config.get('personas', ['none']) if args.interactive or not args.sys_prompt or not args.usr_prompt else ['none'] 
        repetitions = config.get('repetitions', 1) if args.interactive or not args.sys_prompt or not args.usr_prompt else 1
        experiment_folder = config.get('experiment') if args.interactive or not args.sys_prompt or not args.usr_prompt else None
        
        # Set log directory based on selected experiment
        if experiment_folder:
            log_dir_path = Path(f"experiments/{experiment_folder}/logs")
        else:
            log_dir_path = Path(LOG_DIR)
        
        # Generate all experiment combinations
        combinations = list(itertools.product(sys_prompts, usr_prompts, args.models, personas, range(repetitions)))
        
        # For progress tracking, we need to know turns per experiment
        sample_usr_prompt = Path(usr_prompts[0])
        with open(sample_usr_prompt, "r") as f:
            user_prompt_yaml = yaml.safe_load(f)
        num_turns_per_experiment = len(user_prompt_yaml.get("variants", []))
        total_turns = len(combinations) * num_turns_per_experiment
        turn_counter = 0
        lock = threading.Lock()

        # Initialize progress tracking  
        # Inspired by Claude's elegant thinking indicator, dot to bloom to dot progression
        spinner_chars = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✢", "·"]
        spinner_index = 0
        progress_running = True
        
        def update_turn_counter(model_name, model_turn_index, model_output):
            nonlocal turn_counter
            with lock:
                turn_counter += 1

        successes = []  # Will store (model, run_id, log_path) tuples
        failures = []  # Will store (model, error, traceback) tuples
        model_failure_counts = {}  # Track failures per model for systematic detection
        
        def progress_display_worker():
            nonlocal spinner_index
            while progress_running:
                spinner = spinner_chars[spinner_index % len(spinner_chars)]
                error_text = f" ({len(failures)} errors)" if failures else ""
                print(f"\r{spinner} Running... {turn_counter}/{total_turns}{error_text}", end="", flush=True)
                spinner_index += 1
                time.sleep(0.18)  # Update every 180ms for more zen breathing
        
        # Start background progress display
        progress_thread = threading.Thread(target=progress_display_worker, daemon=True)
        progress_thread.start()
        log_dir_path.mkdir(parents=True, exist_ok=True)

        def run_one_experiment(combination):
            sys_prompt, usr_prompt, model, persona, rep = combination
            try:
                result = run_exploit_yaml(
                    yaml_path=str(usr_prompt),
                    sys_prompt=str(sys_prompt),
                    model_name=model,
                    temperature=args.temperature,
                    mode=args.mode,
                    persona_name=persona if persona != "none" else None,
                    drift_threshold=args.drift_threshold,
                    disable_containment=args.disable_containment,
                    experiment_id=args.experiment_id,
                    scenario_hash=args.scenario_hash,
                    score_log=args.score_log,
                    run_command=run_command_str,
                    user_turn_callback=update_turn_counter,
                    experiment_code=args.experiment_code,
                    quiet=True,
                    log_dir=str(log_dir_path),
                )
                with lock:
                    successes.append((f"{model}({rep+1})", result.isbn_run_id, str(log_dir_path / f"{result.isbn_run_id}.json")))
                return result
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                with lock:
                    failures.append((f"{model}({rep+1})", str(e), tb))
                    model_failure_counts[model] = model_failure_counts.get(model, 0) + 1
                return None

        with ThreadPoolExecutor(max_workers=min(len(combinations), 6)) as executor:
            futures = {executor.submit(run_one_experiment, combo): combo for combo in combinations}
            for future in as_completed(futures):
                _ = future.result()
        
        # Stop progress display and clear line
        progress_running = False
        time.sleep(0.6)  # Let the last update finish
        print("\r" + " " * 80 + "\r", end="")

        # Lock in completion state  
        print("⏺ BATCH COMPLETE")
        print("")
        print(f"· Runs completed: {turn_counter}/{total_turns} ({turn_counter/total_turns*100:.0f}%)")
        print(f"· Output directory: {log_dir_path}")
        
        # Show systematic issues if any
        systematic_issues = []
        for model, failure_count in model_failure_counts.items():
            if failure_count >= num_turns_per_model:
                systematic_issues.append(f"{model} (all runs failed)")
        
        if systematic_issues:
            print(f"· Systematic issues: {', '.join(systematic_issues)}")
        
        print("")
        if failures:
            from datetime import datetime
            error_log_path = log_dir_path / "run_failures.txt"
            with open(error_log_path, "w") as f:
                f.write(f"Run state: FAILED\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Command: {run_command_str}\n")
                f.write(f"Failed models: {', '.join(model for model, _, _ in failures)}\n")
                f.write("\n--- Tracebacks ---\n")
                for model, err, tb in failures:
                    f.write(f"\nModel: {model}\nError: {err}\nTraceback:\n{tb}\n{'-'*40}\n")
            print(f"[ERROR] Detailed failure log written to: {error_log_path}")

if __name__ == "__main__":
    main()
