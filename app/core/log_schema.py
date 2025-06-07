from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

# 🧠 Human review data structure
class HumanReview(BaseModel):
    drift_score: Optional[float] = None  # 0.0-1.0, deviation from alignment baseline
    refusal_score: Optional[int] = None  # 1=hard refusal, 2=partial refusal, 3=full compliance
    reviewer: Optional[str] = None  # Initials or username
    review_date: Optional[str] = None  # ISO timestamp
    notes: Optional[str] = None  # Freeform notes about behavior, tone, failure mode
    tags: Optional[List[str]] = []  # Multi-tag list (e.g., ["misinfo", "false_authority", "drift"])

# 🤖 LLM evaluator data structure  
class LLMEvaluation(BaseModel):
    drift_score: Optional[float] = None  # 0.0-1.0, deviation from alignment baseline
    refusal_score: Optional[int] = None  # 1=hard refusal, 2=partial refusal, 3=full compliance
    evaluator_model: Optional[str] = None  # Model used for evaluation (e.g. "claude-3-5-sonnet-20241022")
    evaluation_date: Optional[str] = None  # ISO timestamp
    reasoning: Optional[str] = None  # LLM's explanation of its scoring
    confidence: Optional[float] = None  # 0.0-1.0, LLM's confidence in its assessment

# 🔄 Reproduction information
class ReproductionInfo(BaseModel):
    experiment_code: str  # e.g. "RRS", "GRD"
    model: str  # Canonical model name
    system_prompt: str  # System prompt tag/identifier
    user_prompt: str  # User prompt identifier
    persona: Optional[str] = None  # Persona used, if any
    temperature: Optional[float] = None  # Sampling temperature
    
# 📊 Evaluator version tracking
class EvaluatorVersions(BaseModel):
    red_score_ios: Optional[str] = None  # iOS app version
    llm_evaluator: Optional[str] = None  # LLM model version used for auto-evaluation
    red_core_schema: Optional[str] = None  # Schema version

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
    
    # Dual scoring system
    human_review: Optional[HumanReview] = None  # Human-in-the-loop scoring
    llm_evaluation: Optional[LLMEvaluation] = None  # Automated LLM evaluation
    
    # Token usage
    input_token_count: Optional[int] = None  # Number of input tokens for this turn
    output_token_count: Optional[int] = None  # Number of output tokens for this turn
    total_token_count: Optional[int] = None  # Total tokens (input + output) for this turn
    
    # Legacy fields (deprecated but kept for compatibility)
    drift_score: Optional[float] = None  # DEPRECATED: Use human_review.drift_score
    refusal_score: Optional[int] = None  # DEPRECATED: Use human_review.refusal_score
    notes: Optional[str] = None  # DEPRECATED: Use human_review.notes
    reviewer: Optional[str] = None  # DEPRECATED: Use human_review.reviewer
    tags: Optional[List[str]] = []  # DEPRECATED: Use human_review.tags

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
    
    # Prompt content and references
    system_prompt_tag: str  # Which system prompt was applied (ID tag)
    system_prompt_content: Optional[str] = None  # Full text of system prompt
    system_prompt_file: Optional[str] = None  # Relative path to system prompt file
    system_prompt_hash: str  # SHA-1 of system prompt file contents
    user_prompt_file: Optional[str] = None  # Relative path to user prompt YAML file
    user_prompt_hash: str  # SHA-1 of user prompt YAML contents
    
    # Experimental context
    experiment_readme: Optional[str] = None  # Path to experiment README with methodology
    hypothesis: Optional[str] = None  # Research hypothesis being tested
    
    persona: str  # Persona name in use
    turn_index_offset: Optional[int] = 1  # Usually 1; adjust if multi-file logs are merged
    experiment_id: str  # Short code or name for this experiment
    scenario_hash: str  # Deterministic hash of (prompt + model + temp + persona)
    turns: List[Turn]  # List of conversational turns (the actual data under analysis)
    
    # Enhanced provenance and reproduction
    reproduction_info: Optional[ReproductionInfo] = None  # How to reproduce this experiment
    evaluator_versions: Optional[EvaluatorVersions] = None  # Tool versions used
    red_core_version: Optional[str] = None  # RED_CORE codebase version (Git tag/commit)
    
    # Legacy and metadata
    evaluator_version: Optional[str] = "unknown"  # DEPRECATED: Use evaluator_versions
    review_meta: Optional[Dict[str, Any]] = None  # Top-level review summary block
    run_command: Optional[str] = None  # DEPRECATED: Use reproduction_info
    sdk_version: Optional[str] = None  # Version of the SDK used to run the script
    python_version: Optional[str] = None  # Python version used to run the script
    log_hash: Optional[str] = None  # Unique hash for this log file
    created_at: Optional[str] = None  # ISO timestamp of log creation
    status: Optional[str] = "complete"  # Log status in workflow
    tags: Optional[List[str]] = []  # Custom tags for grouping/filtering
    provenance: Optional[List[Dict[str, Any]]] = []  # History of major events (optional)
    runtime: Optional[str] = None  # OS/platform info

    class Config:
        validate_assignment = True  # Re-validate every assignment
        extra = "forbid"  # Reject any unexpected fields to prevent schema drift
