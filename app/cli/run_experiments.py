import asyncio
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

from .progress_display_async import async_progress_display

from app.core.log_schema import SessionLog, Turn, ReproductionInfo, EvaluatorVersions
from app.core.log_utils import log_session, generate_readable_run_id, get_next_batch_id
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
    list_available_models,
)
from app.core.context import ConversationContext, ConversationHistory
from .experiment_runner_async import run_exploit_yaml_async
from tools.containment import (
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
    elif vendor == "xai":
        from app.api_runners.xai_runner import XAIRunner
        return XAIRunner(model_name=canonical_name)
    elif vendor == "deepseek":
        from app.api_runners.deepseek_runner import DeepSeekRunner
        return DeepSeekRunner(model_name=canonical_name)
    elif vendor == "mistral":
        from app.api_runners.mistral_runner import MistralRunner
        return MistralRunner(model_name=canonical_name)
    elif vendor == "cohere":
        from app.api_runners.cohere_runner import CohereRunner
        return CohereRunner(model_name=canonical_name)
    elif vendor == "together":
        from app.api_runners.together_runner import TogetherRunner
        return TogetherRunner(model_name=canonical_name)
    else:
        raise ValueError(f"No runner registered for model '{model_name}' (resolved as '{canonical_name}')")

def compute_scenario_hash(model_name, system_prompt, user_prompt, temperature, persona=None):
    parts = [model_name, system_prompt, user_prompt, str(temperature), persona or "none"]
    return hash_string("||".join(parts))

def load_persona(persona_name: str) -> dict:
    persona_path = Path("data/prompts/personas") / f"{persona_name}.yaml"
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
    if Path("data/prompts/personas").exists():
        personas = [f.stem for f in Path("data/prompts/personas").glob("*.yaml")]
        return ["none"] + sorted(personas)
    return ["none"]

def select_from_list(console, title, options, allow_multiple=False):
    """Interactive selection with enhanced styling."""
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.panel import Panel
    
    if not options:
        console.print(Panel(f"[red]No {title.lower()} available[/red]", style="red"))
        return None
    
    # Create a styled table for options
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="dim", width=4)
    table.add_column("", style="white")
    
    for i, option in enumerate(options):
        # Get clean display name and styling
        display_name = option
        style = "#c0c0c0"  # Light gray, easier on the eyes
        
        # For file paths, show just the filename
        if "/" in option and option.endswith(".yaml"):
            display_name = Path(option).name
            
            # Auto-assign color by experiment code/pattern using hash
            experiment_colors = [
                "bright_red", "bright_yellow", "bright_blue", "bright_magenta", 
                "bright_cyan", "bright_green", "red", "yellow", "blue", "magenta"
            ]
            
            # Extract experiment identifier for consistent coloring
            exp_id = ""
            if "guardrail_decay" in option or "grd" in option.lower():
                exp_id = "GRD"
            elif "refusal" in option:
                exp_id = "RRS" 
            elif "unicode" in option:
                exp_id = "UNI"
            else:
                # Use directory name as experiment ID
                parts = option.split("/")
                for part in parts:
                    if "experiments" in parts and parts.index(part) > parts.index("experiments"):
                        exp_id = part.upper()[:3]
                        break
                if not exp_id:
                    exp_id = Path(option).parent.name.upper()[:3]
            
            # Assign color based on hash of experiment ID for consistency
            color_index = hash(exp_id) % len(experiment_colors)
            style = experiment_colors[color_index]
        
        # Model type styling (for non-file options)
        elif "gpt" in option.lower():
            style = "bright_green"
        elif "claude" in option.lower():
            style = "bright_magenta" 
        elif "gemini" in option.lower():
            style = "bright_cyan"
        elif "mistral" in option.lower():
            style = "bright_yellow"
        
        table.add_row(f"{i+1}.", f"[{style}]{display_name}[/{style}]")
    
    # Display with panel
    console.print(Panel(
        table, 
        title=f"[bold bright_blue]{title}[/bold bright_blue]",
        border_style="bright_blue"
    ))
    
    if allow_multiple:
        console.print("[dim italic]Enter numbers separated by spaces (e.g., 1 3 5)[/dim italic]")
        while True:
            choice = Prompt.ask("[bold bright_yellow]Select options[/bold bright_yellow]", default="")
            if not choice.strip():
                return []
            try:
                indices = [int(x.strip()) - 1 for x in choice.split()]
                if all(0 <= i < len(options) for i in indices):
                    selected = [options[i] for i in indices]
                    console.print(f"[green]✓ Selected {len(selected)} items[/green]")
                    return selected
                else:
                    console.print("[red]❌ Invalid selection. Please try again.[/red]")
            except ValueError:
                console.print("[red]❌ Please enter numbers only.[/red]")
    else:
        while True:
            choice = Prompt.ask("[bold bright_yellow]Select option[/bold bright_yellow] (number)")
            try:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    console.print(f"[green]✓ Selected: [white]{options[index]}[/white][/green]")
                    return options[index]
                else:
                    console.print(f"[red]❌ Please enter a number between 1 and {len(options)}[/red]")
            except ValueError:
                console.print("[red]❌ Please enter a valid number.[/red]")

def prompt_for_text(console, prompt, required=True):
    """Simple text input prompt."""
    while True:
        value = Prompt.ask(prompt)
        if not required or value.strip():
            return value.strip()
        console.print("[red]This field is required[/red]")

def generate_experiment_code(name: str) -> str:
    """Generate code: first letter + next 2 consonants, uppercase."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    name = name.lower().replace("_", "")
    if not name:
        return "XXX"
    code = name[0]
    for char in name[1:]:
        if char in consonants and len(code) < 3:
            code += char
    while len(code) < 3:
        code += "X"
    return code.upper()


def configure_experiment_interactively():
    """Interactive experiment configuration using Rich with enhanced styling."""
    console = Console()

    # Welcome header
    console.print()
    welcome_text = """[bold bright_cyan]🔬 RED_CORE[/bold bright_cyan] [dim]|[/dim] [bold bright_magenta]Interactive Experiment Configuration[/bold bright_magenta]"""

    console.print(Panel(
        welcome_text,
        style="bright_blue",
        border_style="bright_cyan",
        padding=(0, 2)
    ))
    console.print()

    # 1. Select experiment folder (required)
    experiments = get_available_experiments()
    if not experiments:
        console.print("[red]No experiments found! Create one with: make exp[/red]")
        return None

    experiment = select_from_list(console, "Select Experiment", experiments)
    if experiment is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return None

    # Auto-generate experiment code
    experiment_code = generate_experiment_code(experiment)
    console.print(f"[dim]Experiment code: {experiment_code}[/dim]")
    console.print()

    # 2. System prompts
    sys_prompts_options = get_available_prompts("**/sys*.yaml")
    if not sys_prompts_options:
        console.print("[red]No system prompts found![/red]")
        return None

    sys_prompts = select_from_list(console, "Select System Prompts", sys_prompts_options, allow_multiple=True)
    if not sys_prompts:
        console.print("[yellow]No system prompts selected.[/yellow]")
        return None

    console.print()

    # 3. User prompts (filtered to experiment)
    exp_prompt_dir = f"experiments/{experiment}/prompts"
    usr_prompts_options = get_available_prompts("**/*.yaml", exp_prompt_dir)

    if not usr_prompts_options:
        console.print(f"[yellow]No prompts in {exp_prompt_dir}, searching all experiments...[/yellow]")
        usr_prompts_options = get_available_prompts("**/*.yaml", "experiments/*/prompts")

    if not usr_prompts_options:
        console.print("[red]No user prompts found![/red]")
        return None

    usr_prompts = select_from_list(console, "Select User Prompts", usr_prompts_options, allow_multiple=True)
    if not usr_prompts:
        console.print("[yellow]No user prompts selected.[/yellow]")
        return None

    console.print()

    # 4. Models
    model_choices = list_available_models(include_deprecated=False)
    models = select_from_list(console, "Select Models", model_choices, allow_multiple=True)
    if not models:
        console.print("[yellow]No models selected.[/yellow]")
        return None

    console.print()

    # 5. Personas (Y/N then multi-select if Y)
    personas = ["none"]
    if Confirm.ask("Use personas?", default=False):
        personas_options = get_available_personas()
        if personas_options:
            selected = select_from_list(console, "Select Personas", personas_options, allow_multiple=True)
            if selected:
                personas = selected
        console.print()

    # 6. Repetitions (default 1)
    rep_input = Prompt.ask("Repetitions", default="1")
    repetitions = int(rep_input) if rep_input.isdigit() else 1

    # Temperature fixed at 0.8
    temperature = 0.8

    # Calculate totals
    total_combinations = len(sys_prompts) * len(usr_prompts) * len(models) * len(personas)
    total_runs = total_combinations * repetitions

    # Show summary
    console.print()
    summary_table = Table(title="Configuration Summary", style="cyan")
    summary_table.add_column("Setting", style="white")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Experiment", f"{experiment} [{experiment_code}]")
    summary_table.add_row("System prompts", f"{len(sys_prompts)} files")
    summary_table.add_row("User prompts", f"{len(usr_prompts)} files")
    summary_table.add_row("Models", f"{len(models)} models")
    summary_table.add_row("Personas", f"{len(personas)} variations")
    summary_table.add_row("Repetitions", str(repetitions))
    summary_table.add_row("Temperature", str(temperature))
    summary_table.add_row("[bold]Total runs[/bold]", f"[bold]{total_runs}[/bold]")

    console.print(summary_table)
    console.print()

    if not Confirm.ask("Run?", default=True):
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
        'disable_containment': True,  # Always disabled
        'experiment': experiment
    }

def compute_log_hash(log_dict):
    # Exclude log_hash field itself
    log_copy = copy.deepcopy(log_dict)
    log_copy.pop("log_hash", None)
    return hashlib.sha256(json.dumps(log_copy, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser("new", help="Scaffold a new experiment folder from template.")
    new_parser.add_argument("--name", required=True)
    new_parser.add_argument("--contributors", required=True)
    new_parser.add_argument("--purpose", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiment(s) with specified models and prompts.")
    run_parser.add_argument("--models", nargs="+", default=["gpt-4.1", "claude-3-7-sonnet-20250219", "gemini-2.5-pro-preview-06-05"])
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
    run_parser.add_argument("--evaluator-model", default="gemini-2.0-flash-lite", help="Model to use for LLM evaluation (default: gemini-2.0-flash-lite for cost efficiency)")
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
            
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        console.print()
        
        config_info = f"""[dim]·[/dim] System prompts: [white]{len(sys_prompts)} files[/white]
[dim]·[/dim] User prompts: [white]{len(usr_prompts)} files[/white]
[dim]·[/dim] Models: [bright_cyan]{', '.join(args.models)}[/bright_cyan]"""
        
        if args.disable_containment:
            config_info += "\n[dim]·[/dim] [red]Warning: Containment disabled[/red]"
            
        console.print(Panel(
            config_info,
            title="[bold bright_blue]⚙ BATCH CONFIG[/bold bright_blue]",
            border_style="bright_blue",
            padding=(0, 1)
        ))
        console.print()

        # Capture the command used to run the script
        run_command_str = "PYTHONPATH=. " + " ".join([shlex.quote(arg) for arg in sys.argv])

        # Calculate total experiments and set log directory
        personas = config.get('personas', ['none']) if args.interactive or not args.sys_prompt or not args.usr_prompt else ['none'] 
        repetitions = config.get('repetitions', 1) if args.interactive or not args.sys_prompt or not args.usr_prompt else 1
        experiment_folder = config.get('experiment') if args.interactive or not args.sys_prompt or not args.usr_prompt else None
        
        # Auto-detect experiment codes from directory names
        def get_experiment_code_mapping():
            mapping = {}
            experiments_dir = Path("experiments")
            if experiments_dir.exists():
                for exp_dir in experiments_dir.iterdir():
                    if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                        code = generate_experiment_code(exp_dir.name)
                        mapping[code] = exp_dir.name
            return mapping

        EXPERIMENT_CODE_TO_FOLDER = get_experiment_code_mapping()

        # Set log directory based on selected experiment or experiment code
        if experiment_folder:
            log_dir_path = Path(f"experiments/{experiment_folder}/logs")
        elif args.experiment_code and args.experiment_code.upper()[:3] in EXPERIMENT_CODE_TO_FOLDER:
            mapped_folder = EXPERIMENT_CODE_TO_FOLDER[args.experiment_code.upper()[:3]]
            log_dir_path = Path(f"experiments/{mapped_folder}/logs")
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
        
        # Calculate expected experiments per model for systematic failure detection
        experiments_per_model = len(sys_prompts) * len(usr_prompts) * len(personas) * repetitions
        num_turns_per_model = experiments_per_model * num_turns_per_experiment
        
        turn_counter = 0
        lock = threading.Lock()
        successes = []  # Will store (model, run_id, log_path) tuples
        failures = []  # Will store (model, error, traceback) tuples
        model_failure_counts = {}  # Track failures per model for systematic detection
        
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Use async progress display for proper async/await integration
        async def run_experiments_async():
            async with async_progress_display(total_turns) as (update_counter, add_error):
                
                async def update_turn_counter(model_name, model_turn_index, model_output):
                    """Updated async callback to work with new progress display"""
                    nonlocal turn_counter
                    turn_counter += 1
                    await update_counter(model_name, model_turn_index, model_output)

                async def run_one_experiment(combination):
                    sys_prompt, usr_prompt, model, persona, rep = combination
                    try:
                        # PHASE R1: Use clean async architecture - no more asyncio.run()
                        result = await run_exploit_yaml_async(
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
                        successes.append((f"{model}({rep+1})", result.isbn_run_id, str(log_dir_path / f"{result.isbn_run_id}.json")))
                        return result
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        failures.append((f"{model}({rep+1})", str(e), tb))
                        model_failure_counts[model] = model_failure_counts.get(model, 0) + 1
                        await add_error(f"{model}({rep+1})", str(e))  # Track error in progress display
                        return None

                # Run experiments concurrently with asyncio.gather instead of ThreadPoolExecutor
                tasks = [run_one_experiment(combo) for combo in combinations]
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute the async function
        asyncio.run(run_experiments_async())
        
        # Rich Live automatically handles cleanup - no manual clearing needed!

        # Lock in completion state with golden finish
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        
        completion_rate = turn_counter/total_turns*100 if total_turns > 0 else 0
        
        # Choose completion color based on success rate
        if completion_rate >= 95:
            completion_color = "#228b22"  # Forest green for high success
            status_text = "BATCH COMPLETE"
        elif completion_rate >= 80:
            completion_color = "#ffa500"  # Orange for moderate success  
            status_text = "BATCH COMPLETE"
        else:
            completion_color = "#dc143c"  # Crimson for low success
            status_text = "BATCH COMPLETE"
            
        completion_text = f"""[bold {completion_color}]{status_text}[/bold {completion_color}]

[dim]·[/dim] Runs completed: [bold white]{turn_counter}/{total_turns}[/bold white] [dim]([white]{completion_rate:.0f}%[/white])[/dim]
[dim]·[/dim] Output directory: [bold cyan]{log_dir_path}[/bold cyan]"""

        console.print(Panel(
            completion_text,
            border_style="dim",
            padding=(0, 1)
        ))
        
        # Automatically run dual evaluation (always enabled in production)
        if successes:
            console.print()
            console.print("[bold bright_cyan]🎯 Starting Automated Dual Evaluation[/bold bright_cyan]")
            
            try:
                from app.analysis.dual_evaluator import DualEvaluator
                
                async def run_dual_evaluation():
                    evaluator = DualEvaluator(args.evaluator_model)
                    results = await evaluator.batch_evaluate_directory(log_dir_path)
                    return results
                
                # Run dual evaluation
                eval_results = asyncio.run(run_dual_evaluation())
                
                if eval_results:
                    console.print(f"[green]✓ Completed dual evaluation of {len(eval_results)} log files[/green]")
                    
                    # Show summary of evaluation results
                    total_agreement = 0
                    total_confidence = 0
                    for result in eval_results:
                        comparison = result.get("scoring_comparison", {})
                        agreement = comparison.get("method_agreement", {}).get("refusal_scoring", 0)
                        if agreement:
                            total_agreement += agreement
                        
                        llm_eval = result.get("llm_evaluation", {})
                        confidence = llm_eval.get("aggregate_metrics", {}).get("average_confidence", 0)
                        if confidence:
                            total_confidence += confidence
                    
                    avg_agreement = total_agreement / len(eval_results) if eval_results else 0
                    avg_confidence = total_confidence / len(eval_results) if eval_results else 0
                    
                    console.print(f"[dim]·[/dim] Method agreement: [bold white]{avg_agreement:.1%}[/bold white]")
                    console.print(f"[dim]·[/dim] LLM confidence: [bold white]{avg_confidence:.1%}[/bold white]")
                    console.print(f"[dim]·[/dim] Evaluation model: [bold cyan]{args.evaluator_model}[/bold cyan]")
                    
                    # Show dual evaluation output directory
                    dual_eval_dir = log_dir_path.parent / "dual_evaluated"
                    if dual_eval_dir.exists():
                        console.print(f"[dim]·[/dim] Analysis files: [bold cyan]{dual_eval_dir}[/bold cyan]")
                else:
                    console.print("[yellow]⚠ Dual evaluation completed but no results generated[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]❌ Dual evaluation failed: {e}[/red]")
                console.print("[dim]Experiment logs saved but scoring incomplete[/dim]")
        
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
