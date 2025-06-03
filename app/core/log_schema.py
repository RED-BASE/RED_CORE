from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

# 🚨 One full conversational turn: prompt -> model response
class Turn(BaseModel):
    turn_index: int  # Order of this turn in the conversation (starts from offset)
    user_input_id: Optional[str] = None  # ID of the prompt variant for this turn
    persona: str  # Active persona used (e.g. "malicious_actor", "none")
    raw_user_input: str  # Original raw user input (may differ from rendered_user_input if transformed)
    rendered_user_input: str  # The final prompt shown to the model after all processing
    model_output: str  # The LLM's full response
    latency_ms: Optional[float] = None  # Time in milliseconds from sending user input to receiving model output
    containment_flags: Optional[List[str]] = []  # Safety flags triggered by this turn, if any
    containment_summary: Optional[Dict[str, Any]] = {}  # Detailed containment data (regexes, rule names, etc.)
    review_status: Optional[str] = "pending"  # Review workflow: pending / flagged / cleared
    drift_score: Optional[float] = None  # Score measuring deviation from alignment baseline (computed later)
    refusal_score: Optional[int] = None  # 1 = full refusal, 2 = partial refusal, 3 = no refusal
    notes: Optional[str] = None  # Freeform notes about behavior, tone, failure mode
    reviewer: Optional[str] = None  # Initials or username
    tags: Optional[List[str]] = []  # Multi-tag list (e.g., ["misinfo", "false_authority", "drift"])
    input_token_count: Optional[int] = None  # Number of input tokens for this turn
    output_token_count: Optional[int] = None  # Number of output tokens for this turn
    total_token_count: Optional[int] = None  # Total tokens (input + output) for this turn
   

    class Config:
        validate_assignment = True  # Will re-check types on every assignment
        extra = "forbid"  # Disallow any fields not explicitly defined above

# 📦 One full conversation session log, with many turns and metadata
class SessionLog(BaseModel):
    isbn_run_id: str  # Unique ID for this log (includes model, prompt tag, timestamp, hash)
    exploit_path: str  # Filepath to the user YAML that generated these prompts
    model: str  # Canonical model name (e.g. "gpt-4o")
    model_vendor: str  # Organization (e.g. openai, anthropic)
    mode: Literal["audit", "attack", "consistency", "compare", "replay"]  # Experiment type
    temperature: float  # Sampling temperature used
    system_prompt_tag: str  # Which system prompt was applied (ID tag)
    system_prompt_hash: str  # SHA-1 of system prompt file contents
    user_prompt_hash: str  # SHA-1 of user prompt YAML contents
    persona: str  # Persona name in use
    turn_index_offset: Optional[int] = 1  # Usually 1; adjust if multi-file logs are merged
    experiment_id: str  # Short code or name for this experiment
    scenario_hash: str  # Deterministic hash of (prompt + model + temp + persona)
    turns: List[Turn]  # List of conversational turns (the actual data under analysis)
    evaluator_version: Optional[str] = "unknown"  # Version tag of the evaluation/scoring tool (if used)
    review_meta: Optional[Dict[str, Any]] = None  # Top-level review summary block
    run_command: Optional[str] = None  # Command used to run the test for reproducibility
    sdk_version: Optional[str] = None  # Version of the SDK used to run the script
    python_version: Optional[str] = None  # Python version used to run the script
    log_hash: Optional[str] = None  # Unique hash for this log file
    red_core_version: Optional[str] = None  # RED_CORE codebase version
    created_at: Optional[str] = None  # ISO timestamp of log creation
    status: Optional[str] = "complete"  # Log status in workflow
    tags: Optional[List[str]] = []  # Custom tags for grouping/filtering
    provenance: Optional[List[Dict[str, Any]]] = []  # History of major events (optional)
    runtime: Optional[str] = None  # OS/platform info

    class Config:
        validate_assignment = True  # Re-validate every assignment
        extra = "forbid"  # Reject any unexpected fields to prevent schema drift
