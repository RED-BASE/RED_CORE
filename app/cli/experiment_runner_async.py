"""
Modern async experiment runner - REFACTORED VERSION

This replaces the massive 248-line run_exploit_yaml() function with clean,
testable components using dependency injection and async/await patterns.

Key improvements:
- Single responsibility components instead of god function
- Dependency injection for testability  
- Async/await for proper concurrency
- Immutable configuration objects
- Comprehensive error handling
- Type safety throughout
"""

import asyncio
import hashlib
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import openai
import yaml
from app.config import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_EXPERIMENT_CODE,
    DEFAULT_MODE,
    DEFAULT_TEMPERATURE,
    get_model_code,
    get_model_snapshot_id,
    get_model_vendor,
    resolve_model,
)
from app.core.context import ConversationContext, ConversationHistory
from app.core.hash_utils import hash_string
from app.core.log_schema import (
    EvaluatorVersions,
    ReproductionInfo,
    SessionLog,
    Turn,
)
from app.core.log_utils import generate_readable_run_id, get_next_batch_id, log_session
from app.core.logger import get_experiment_logger
from tools.containment import (
    containment_summary,
    flatten_containment_flags,
    override_output_if_flagged,
)


# Protocol for API runners to ensure type safety
class APIRunner(Protocol):
    """Protocol defining the interface for AI model runners."""
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from the model."""
        ...
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the runner."""
        ...
    
    def set_persona(self, persona: Dict[str, Any]) -> None:
        """Set the persona configuration for the runner."""
        ...


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""
    yaml_path: Path
    sys_prompt_path: Path
    model_name: str
    temperature: float = DEFAULT_TEMPERATURE
    mode: str = DEFAULT_MODE
    persona_name: Optional[str] = None
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD
    disable_containment: bool = False
    experiment_id: Optional[str] = None
    scenario_hash: Optional[str] = None
    score_log: bool = False
    run_command: Optional[str] = None
    experiment_code: str = DEFAULT_EXPERIMENT_CODE
    quiet: bool = False
    log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
        if not self.sys_prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {self.sys_prompt_path}")


@dataclass(frozen=True)
class PromptData:
    """Immutable prompt data extracted from YAML files."""
    user_prompt_variants: List[Dict[str, Any]]
    user_prompt_hash: str
    user_prompt_raw: str
    system_prompt_content: str
    system_prompt_hash: str
    system_prompt_shorthand: str
    system_prompt_tag: str
    hypothesis: Optional[str] = None


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable model metadata."""
    canonical_name: str
    model_code: str
    vendor: str
    snapshot_id: str


@dataclass
class ExperimentResult:
    """Mutable result object for experiment execution."""
    session_log: SessionLog
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: float = 0


class PromptLoader:
    """Handles loading and parsing of prompt files."""
    
    @staticmethod
    def load_prompt_data(config: ExperimentConfig) -> PromptData:
        """Load and parse prompt data from configuration files."""
        # Load user prompt YAML
        user_prompt_raw = config.yaml_path.read_text()
        user_prompt_data = yaml.safe_load(user_prompt_raw)
        user_prompt_hash = hash_string(user_prompt_raw)
        variants = user_prompt_data.get("variants", [])
        hypothesis = user_prompt_data.get("hypothesis")
        
        # Load system prompt YAML
        sys_prompt_raw = config.sys_prompt_path.read_text()
        sys_prompt_data = yaml.safe_load(sys_prompt_raw)
        sys_prompt_hash = hash_string(sys_prompt_raw)
        
        system_prompt_content = sys_prompt_data.get('system_prompt', '').strip()
        system_prompt_shorthand = sys_prompt_data.get("shorthand", "")
        system_prompt_tag = config.sys_prompt_path.stem + ":latest"
        
        return PromptData(
            user_prompt_variants=variants,
            user_prompt_hash=user_prompt_hash,
            user_prompt_raw=user_prompt_raw,
            system_prompt_content=system_prompt_content,
            system_prompt_hash=sys_prompt_hash,
            system_prompt_shorthand=system_prompt_shorthand,
            system_prompt_tag=system_prompt_tag,
            hypothesis=hypothesis
        )


class ModelMetadataService:
    """Handles model metadata resolution and validation."""
    
    @staticmethod
    def resolve_model_metadata(model_name: str) -> ModelMetadata:
        """Resolve model metadata from model name."""
        canonical_name = resolve_model(model_name)
        model_code = get_model_code(canonical_name)
        vendor = get_model_vendor(canonical_name)
        snapshot_id = get_model_snapshot_id(vendor, canonical_name)
        
        return ModelMetadata(
            canonical_name=canonical_name,
            model_code=model_code,
            vendor=vendor,
            snapshot_id=snapshot_id
        )


class SessionLogFactory:
    """Creates SessionLog objects with proper metadata."""
    
    @staticmethod
    def create_session_log(
        config: ExperimentConfig,
        prompt_data: PromptData,
        model_metadata: ModelMetadata,
        batch_id: str
    ) -> SessionLog:
        """Create a properly configured SessionLog object."""
        
        # Map experiment codes to folders
        EXPERIMENT_CODE_TO_FOLDER = {
            "80K": "80k_hours_demo",
            "GRD": "guardrail_decay", 
            "RRS": "refusal_robustness",
            "UEX": "universal_exploits",
            "DMO": "demo"
        }
        experiment_folder = EXPERIMENT_CODE_TO_FOLDER.get(
            config.experiment_code, 
            config.experiment_code.lower()
        )
        
        return SessionLog(
            isbn_run_id=generate_readable_run_id(
                model_name=model_metadata.canonical_name,
                user_prompt_tag=config.yaml_path.stem,
                system_prompt_tag=prompt_data.system_prompt_tag,
                persona=config.persona_name,
                experiment_code=config.experiment_code,
                system_prompt_shorthand=prompt_data.system_prompt_shorthand,
            ),
            exploit_path=str(config.yaml_path),
            model=model_metadata.canonical_name,
            model_vendor=model_metadata.vendor,
            mode=config.mode,
            temperature=config.temperature,
            
            # Enhanced prompt content and references
            system_prompt_tag=prompt_data.system_prompt_tag,
            system_prompt_content=prompt_data.system_prompt_content,
            system_prompt_file=str(config.sys_prompt_path.resolve().relative_to(Path.cwd().resolve())),
            system_prompt_hash=prompt_data.system_prompt_hash,
            user_prompt_file=str(config.yaml_path.resolve().relative_to(Path.cwd().resolve())),
            user_prompt_hash=prompt_data.user_prompt_hash,
            
            # Experimental context
            experiment_readme=f"experiments/{config.experiment_code.lower()}/README.md" if config.experiment_code else None,
            hypothesis=prompt_data.hypothesis,
            
            persona=config.persona_name or "none",
            turn_index_offset=1,
            experiment_id=config.experiment_id or config.yaml_path.stem,
            scenario_hash=config.scenario_hash or SessionLogFactory._compute_scenario_hash(
                model_metadata.canonical_name,
                prompt_data.system_prompt_content,
                prompt_data.user_prompt_raw,
                config.temperature,
                config.persona_name
            ),
            turns=[],
            
            # Enhanced provenance tracking
            reproduction_info=ReproductionInfo(
                experiment_code=config.experiment_code,
                model=model_metadata.canonical_name,
                system_prompt=prompt_data.system_prompt_tag,
                user_prompt=config.yaml_path.stem,
                persona=config.persona_name,
                temperature=config.temperature
            ),
            evaluator_versions=EvaluatorVersions(
                red_core_schema="2.0.0",
                llm_evaluator=None
            ),
            red_core_version=SessionLogFactory._get_red_core_version(),
            
            # Legacy compatibility fields
            evaluator_version="unknown",
            run_command=config.run_command,
            sdk_version=openai.__version__,
            python_version=sys.version,
            created_at=datetime.now().isoformat(),
            status="complete",
            tags=[],
            provenance=[],
            runtime=platform.platform(),
            
            # Workflow state tracking
            workflow={
                "batch_id": batch_id,
                "batch_created": datetime.now().isoformat(),
                "experiment_name": experiment_folder,
                "evaluations": {
                    "llm": {
                        "completed": False,
                        "date": None,
                        "model": None,
                        "error": None
                    }
                }
            }
        )
    
    @staticmethod
    def _compute_scenario_hash(
        model_name: str, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float,
        persona: Optional[str]
    ) -> str:
        """Compute scenario hash for reproducibility."""
        parts = [model_name, system_prompt, user_prompt, str(temperature), persona or "none"]
        return hash_string("||".join(parts))
    
    @staticmethod
    def _get_red_core_version() -> str:
        """Get RED_CORE version."""
        # This would normally read from a version file or package metadata
        return "3.0.0-refactor"


class TurnProcessor:
    """Processes individual turns in an experiment."""
    
    def __init__(self, runner: APIRunner):
        self.runner = runner
        self.logger = get_experiment_logger("turn_processor")
    
    async def process_turn(
        self,
        variant: Dict[str, Any],
        turn_index: int,
        history: ConversationHistory,
        config: ExperimentConfig,
        prompt_data: PromptData,
        model_metadata: ModelMetadata,
        turn_callback: Optional[Callable] = None
    ) -> Turn:
        """Process a single turn of the experiment."""
        
        # Extract and clean prompt
        raw_prompt = variant.get("prompt", "")
        if not raw_prompt.strip():
            raise ValueError(f"Empty prompt in variant: {variant.get('id', '[no id]')}")
        
        # Parse prompt header and body
        if "\n" in raw_prompt:
            header, body = raw_prompt.split("\n", 1)
        else:
            header, body = "", raw_prompt
        
        prompt_body = body.strip()
        if prompt_body.startswith('"') and prompt_body.endswith('"'):
            prompt_body = prompt_body[1:-1]  # Remove YAML literal quotes
        
        # Create conversation context
        ctx = ConversationContext(
            rendered_prompt=prompt_body,
            persona=config.persona_name or "none",
            system_prompt_tag=prompt_data.system_prompt_tag,
            meta={
                "variant_id": variant.get("id"),
                "prompt_header": header.strip(),
            },
        )
        ctx.user_input = prompt_body
        history.append_user(prompt_body)
        
        # Generate model response
        generate_kwargs = {
            "temperature": config.temperature,
            "turn_index": turn_index,
        }
        if model_metadata.vendor in {"google", "openai"}:
            generate_kwargs["conversation"] = history
        
        start_time = time.time()
        result = self.runner.generate(prompt_body, **generate_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        # Update context with result
        ctx.update_output(result["model_output"])
        ctx.meta.update({
            "vendor_model_id": result.get("model_name"),
            "usage": result.get("usage"),
        })
        history.append_assistant(ctx.model_output)
        
        # Apply containment if enabled
        summary = containment_summary(ctx.user_input, ctx.rendered_prompt, ctx.model_output)
        flags = flatten_containment_flags(summary)
        if flags and not config.disable_containment:
            ctx.update_output(override_output_if_flagged(ctx.model_output, flags))
        
        # Create turn object
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
        
        # Call progress callback if provided
        if turn_callback:
            await turn_callback(model_metadata.canonical_name, turn_index, ctx.model_output)
        
        return turn_obj


class AsyncExperimentRunner:
    """
    Main experiment runner using clean async architecture.
    
    This replaces the massive 248-line run_exploit_yaml() function with
    clean, testable components using dependency injection.
    """
    
    def __init__(
        self,
        prompt_loader: PromptLoader,
        model_service: ModelMetadataService,
        session_factory: SessionLogFactory,
        get_runner_func: Callable[[str], APIRunner],
        get_persona_func: Optional[Callable[[str], Dict[str, Any]]] = None
    ):
        self.prompt_loader = prompt_loader
        self.model_service = model_service
        self.session_factory = session_factory
        self.get_runner_func = get_runner_func
        self.get_persona_func = get_persona_func
        self.logger = get_experiment_logger("async_runner")
    
    async def run_experiment(
        self,
        config: ExperimentConfig,
        turn_callback: Optional[Callable] = None
    ) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        This is the clean, async replacement for the massive run_exploit_yaml() function.
        """
        start_time = time.time()
        
        try:
            # Load and validate all data
            prompt_data = self.prompt_loader.load_prompt_data(config)
            model_metadata = self.model_service.resolve_model_metadata(config.model_name)
            
            # Initialize AI runner and set configuration
            runner = self.get_runner_func(model_metadata.canonical_name)
            if hasattr(runner, "set_system_prompt"):
                runner.set_system_prompt(prompt_data.system_prompt_content)
            
            # Load persona if specified
            if config.persona_name and self.get_persona_func:
                persona_data = self.get_persona_func(config.persona_name)
                if hasattr(runner, "set_persona"):
                    runner.set_persona(persona_data)
            
            # Generate batch ID for tracking
            EXPERIMENT_CODE_TO_FOLDER = {
                "80K": "80k_hours_demo",
                "GRD": "guardrail_decay", 
                "RRS": "refusal_robustness",
                "UEX": "universal_exploits",
                "DMO": "demo"
            }
            experiment_folder = EXPERIMENT_CODE_TO_FOLDER.get(
                config.experiment_code, 
                config.experiment_code.lower()
            )
            batch_id = get_next_batch_id(experiment_folder)
            
            # Create session log
            session_log = self.session_factory.create_session_log(
                config, prompt_data, model_metadata, batch_id
            )
            
            # Initialize conversation history and turn processor
            history = ConversationHistory(system_prompt=prompt_data.system_prompt_content)
            turn_processor = TurnProcessor(runner)
            
            # Process all turns
            for i, variant in enumerate(prompt_data.user_prompt_variants):
                turn_index = len(session_log.turns) + session_log.turn_index_offset
                
                turn = await turn_processor.process_turn(
                    variant=variant,
                    turn_index=turn_index,
                    history=history,
                    config=config,
                    prompt_data=prompt_data,
                    model_metadata=model_metadata,
                    turn_callback=turn_callback
                )
                
                session_log.turns.append(turn)
            
            # Finalize and save the log
            await self._finalize_and_save_log(session_log, config)
            
            execution_time = (time.time() - start_time) * 1000
            return ExperimentResult(
                session_log=session_log,
                success=True,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Experiment failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ExperimentResult(
                session_log=SessionLog(),  # Empty log for failed experiment
                success=False,
                error_message=error_msg,
                execution_time_ms=execution_time
            )
    
    async def _finalize_and_save_log(self, session_log: SessionLog, config: ExperimentConfig) -> None:
        """Finalize log with hash and save to disk."""
        # Compute log hash
        log_dict = session_log.model_dump()
        log_hash = self._compute_log_hash(log_dict)
        session_log.log_hash = log_hash
        
        # Determine log path
        if config.log_dir:
            log_path = config.log_dir / f"{session_log.isbn_run_id}.json"
        else:
            from app.config import LOG_DIR
            log_path = Path(LOG_DIR) / f"{session_log.isbn_run_id}.json"
        
        # Save log
        log_session(str(log_path), session_log)
        
        if not config.quiet:
            self.logger.info(f"Log saved to: {log_path}")
    
    @staticmethod
    def _compute_log_hash(log_dict: Dict[str, Any]) -> str:
        """Compute hash of the log for integrity verification."""
        import json
        log_json = json.dumps(log_dict, sort_keys=True)
        return hashlib.sha256(log_json.encode()).hexdigest()


# Factory function for creating experiment runner with dependencies
def create_experiment_runner() -> AsyncExperimentRunner:
    """Create experiment runner with all dependencies injected."""
    from app.cli.run_experiments import get_runner, load_persona
    
    return AsyncExperimentRunner(
        prompt_loader=PromptLoader(),
        model_service=ModelMetadataService(),
        session_factory=SessionLogFactory(),
        get_runner_func=get_runner,
        get_persona_func=load_persona
    )


# Backward compatibility function matching original signature
async def run_exploit_yaml_async(
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
) -> SessionLog:
    """
    Async replacement for the original run_exploit_yaml() function.
    
    Maintains the same API for backward compatibility while using clean
    async architecture internally.
    """
    config = ExperimentConfig(
        yaml_path=Path(yaml_path),
        sys_prompt_path=Path(sys_prompt),
        model_name=model_name,
        temperature=temperature,
        mode=mode,
        persona_name=persona_name,
        drift_threshold=drift_threshold,
        disable_containment=disable_containment,
        experiment_id=experiment_id,
        scenario_hash=scenario_hash,
        score_log=score_log,
        run_command=run_command,
        experiment_code=experiment_code,
        quiet=quiet,
        log_dir=Path(log_dir) if log_dir else None
    )
    
    # Convert sync callback to async if provided
    async_callback = None
    if user_turn_callback:
        import asyncio
        async def async_wrapper(model_name: str, turn_index: int, model_output: str):
            # Check if callback is already async
            if asyncio.iscoroutinefunction(user_turn_callback):
                await user_turn_callback(model_name, turn_index, model_output)
            else:
                user_turn_callback(model_name, turn_index, model_output)
        async_callback = async_wrapper
    
    runner = create_experiment_runner()
    result = await runner.run_experiment(config, async_callback)
    
    if not result.success:
        raise RuntimeError(result.error_message)
    
    return result.session_log