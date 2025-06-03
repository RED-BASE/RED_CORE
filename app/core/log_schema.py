from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

# 🚨 One full conversational turn: prompt -> model response
class Turn(BaseModel):
    turn_index: int  # Order of this turn in the conversation (starts from offset)
    rendered_prompt: str  # What the model was actually shown this turn
    user_input: str  # Original raw user input (may differ from rendered_prompt if transformed)
    model_output: str  # The LLM's full response
    persona: str  # Active persona used (e.g. "malicious_actor", "none")
    system_prompt_tag: str  # ID of the system prompt in use (e.g. "sys_guardrail_decay_01:latest")
    meta: Optional[Dict[str, Any]] = {}  # Variant ID, headers, or any injected metadata
    system_prompt_text: Optional[str] = None  # Full text of system prompt used (captured here for auditability)
    containment_flags: Optional[List[str]] = []  # Safety flags triggered by this turn, if any
    containment_summary: Optional[Dict[str, Any]] = {}  # Detailed containment data (regexes, rule names, etc.)
    drift_score: Optional[float] = None  # Score measuring deviation from alignment baseline (computed later)
    drift_notes: Optional[str] = None  # Human or model-generated notes on why this turn shows drift
    review_status: Optional[str] = "pending"  # Review workflow: pending / flagged / cleared
    prompt_hash: Optional[str] = None # hash of the prompt used
    completion_hash: Optional[str] = None # hash of the completion used
    sdk_version: Optional[str] = None # version of the sdk used to run the script
    python_version: Optional[str] = None # python version used to run the script
    model_version: Optional[str] = None # version of the model used to run the script
    model_vendor: Optional[str] = None # vendor of the model used to run the script
    model_name: Optional[str] = None # name of the model used to run the script
    model_temperature: Optional[float] = None # temperature of the model used to run the script
    model_top_p: Optional[float] = None # top_p of the model used to run the script
    model_top_k: Optional[int] = None # top_k of the model used to run the script
    model_max_tokens: Optional[int] = None # max_tokens of the model used to run the script

    # Per-Turn Evaluation Fields
    refusal_score: Optional[int] = None  # 1 = full refusal, 2 = partial refusal, 3 = no refusal
    tags: Optional[List[str]] = []  # Multi-tag list (e.g., ["misinfo", "false_authority", "drift"])
    notes: Optional[str] = None  # Freeform notes about behavior, tone, failure mode
    reviewer: Optional[str] = None  # Initials or username
    timestamp: Optional[str] = None  # ISO format time of scoring

    class Config:
        validate_assignment = True  # Will re-check types on every assignment
        extra = "forbid"  # Disallow any fields not explicitly defined above

# 📦 One full conversation session log, with many turns and metadata
class SessionLog(BaseModel):
    isbn_run_id: str  # Unique ID for this log (includes model, prompt tag, timestamp, hash)
    exploit_path: str  # Filepath to the user YAML that generated these prompts
    model: str  # Canonical model name (e.g. "gpt-4o")
    model_code: str  # Short code for this model (e.g. "GPT4O")
    model_vendor: str  # Organization (e.g. openai, anthropic)
    model_snapshot_id: Optional[str] = None  # Specific release/snapshot (e.g. "gpt-4o-2025-05-26")
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

    class Config:
        validate_assignment = True  # Re-validate every assignment
        extra = "forbid"  # Reject any unexpected fields to prevent schema drift
