from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from datetime import datetime

class EvaluationResult(BaseModel):
    score: Optional[float] = None
    criteria: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    consistency_score: Optional[float] = None 
    unique_responses_count: Optional[int] = None
    difference: Optional[bool] = None
    summary: Optional[str] = None

class DefenseDetails(BaseModel):
    prefix_length: Optional[int] = None
    args: Optional[Dict[str, Any]] = None

class ComparisonDelta(BaseModel):
    output_changed: Optional[bool] = None
    error_status_changed: Optional[bool] = None
    error_change_type: Optional[Literal['cleared', 'introduced', 'persisted']] = None
    score_calculated: Optional[bool] = None
    score_delta: Optional[float] = None
    score_change_significant: Optional[bool] = None
    criteria_comparison: Optional[Dict[str, Any]] = None

class UnifiedEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    run_id: str
    config_hash: str
    mode: Literal["attack", "consistency", "compare", "replay"]
    run_label: Optional[str] = None
    experiment_id: str
    prompt_type: Optional[str] = None
    repetition: Optional[int] = None
    turn_index: Optional[int] = None
    model: Optional[str] = None
    model_a: Optional[str] = None
    model_b: Optional[str] = None
    base_prompt: Optional[str] = None
    attack_prompt: Optional[str] = None
    original_effective_prompt: Optional[str] = None
    replay_effective_prompt: Optional[str] = None
    user_prompt_hash: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    guardrail_version: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    output_a: Optional[str] = None
    output_b: Optional[str] = None
    evaluator: Optional[EvaluationResult] = None
    evaluator_type: Optional[str] = None
    eval_config: Optional[Dict[str, Any]] = None
    scoring_summary: Optional[Dict[str, float]] = None
    drift_label: Optional[str] = None
    drift_notes: Optional[str] = None
    defense_applied: Optional[bool] = None
    defense: Optional[str] = "none"
    defense_args: Optional[Dict[str, Any]] = {}
    defense_details: Optional[DefenseDetails] = None
    comparison_delta: Optional[ComparisonDelta] = None
    source_log_file: Optional[str] = None

    class Config:
        validate_assignment = True
        extra = 'ignore'
