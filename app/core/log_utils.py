# log_utils.py

from app.core.log_schema import SessionLog
from pydantic import ValidationError
import json
from pathlib import Path
from datetime import datetime
import hashlib
import fcntl
import os
import time
from app.config import get_model_code
from typing import Optional


def generate_readable_run_id(
    model_name: str,
    user_prompt_tag: str,
    system_prompt_tag: Optional[str] = None,
    persona: Optional[str] = None,
    experiment_code: str = "GRD",
    system_prompt_shorthand: Optional[str] = None
) -> str:
    """Create a readable unique run ID for naming logs."""
    model_code = get_model_code(model_name)

    def normalize_tag(tag: str, prefix: str) -> str:
        if not tag:
            return ""
        tag = tag.lower().strip()
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
        digits = "".join(filter(str.isdigit, tag))[:2]
        return f"{prefix.upper()}{digits.zfill(2)}" if digits else ""

    # Use shorthand if provided, otherwise fall back to tag extraction
    if system_prompt_shorthand:
        s_tag = system_prompt_shorthand.upper()
    else:
        s_tag = normalize_tag(system_prompt_tag, "sys")
    
    u_tag = normalize_tag(user_prompt_tag, "usr")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    # Include microseconds in hash to ensure uniqueness for rapid batch runs
    unique = hashlib.sha1(f"{model_name}{system_prompt_tag}{user_prompt_tag}{persona}{datetime.now().isoformat()}".encode()).hexdigest()[:6].upper()

    parts = [experiment_code, model_code]
    if s_tag: parts.append(s_tag)
    if u_tag: parts.append(u_tag)
    if persona: parts.append(persona.upper())
    parts.extend([timestamp, unique])

    return "-".join(parts)



def log_session(log_filepath: str, session_log: SessionLog):
    """
    Save a fully validated SessionLog object to disk as JSON.
    """
    json_string = session_log.model_dump_json(indent=2)
    Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(json_string)

def get_next_batch_id(experiment_name: str, purpose: Optional[str] = None) -> str:
    """
    Get the next batch ID for an experiment, with race condition protection.
    
    Uses file locking to ensure atomic read-increment-write operations
    when multiple experiments run simultaneously. Enhanced for multi-user
    scenarios with better error handling and user identification.
    
    Args:
        experiment_name: Name of the experiment (e.g., "demo", "refusal_robustness")
        purpose: Optional description of batch purpose for tracking
    
    Returns:
        Batch ID in format "{experiment_name}-{count:02d}" (e.g., "demo-03")
    """
    from pathlib import Path
    import os
    import tempfile
    import time
    
    # Use configurable experiments directory
    experiments_dir = Path(os.getenv("RED_CORE_EXPERIMENTS_DIR", "experiments"))
    experiments_dir.mkdir(exist_ok=True)
    
    experiment_dir = experiments_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    counter_file = experiment_dir / ".batch_counter"
    lock_file = experiment_dir / ".batch_counter.lock"
    
    # Multi-user safety: include user info for debugging
    user_info = f"user:{os.getenv('USER', 'unknown')}_pid:{os.getpid()}"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Use file locking to prevent race conditions
            with open(counter_file, "a+") as f:
                # Lock the file for exclusive access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # Read current count
                f.seek(0)
                content = f.read().strip()
                
                if content:
                    try:
                        count = int(content) + 1
                    except ValueError:
                        # Handle corrupted counter file
                        print(f"⚠️  Corrupted batch counter for {experiment_name}, resetting to 1")
                        count = 1
                else:
                    count = 1
                
                # Write new count with metadata
                f.seek(0)
                f.truncate()
                f.write(f"{count}\n# Last updated by {user_info} at {datetime.now().isoformat()}")
                if purpose:
                    f.write(f"\n# Purpose: {purpose}")
                
                # Lock is automatically released when file closes
            
            batch_id = f"{experiment_name}-{count:02d}"
            
            # Log batch creation for debugging
            print(f"📦 Created batch {batch_id}" + (f" - {purpose}" if purpose else ""))
            
            return batch_id
            
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                # Brief retry delay with jitter
                time.sleep(0.1 + (attempt * 0.1))
                continue
            else:
                # Fallback to timestamp-based ID for reliability
                timestamp = datetime.now().strftime("%m%d_%H%M%S")
                fallback_id = f"{experiment_name}-{timestamp}"
                print(f"⚠️  Batch counter failed for {experiment_name}, using fallback: {fallback_id}")
                print(f"Error: {e}")
                return fallback_id
    
    # Should never reach here, but safety fallback
    return f"{experiment_name}-emergency"


def needs_llm_evaluation(log_data: dict) -> bool:
    """
    Check if a log needs LLM evaluation based on workflow state.
    
    Args:
        log_data: Parsed log JSON data
    
    Returns:
        True if log needs LLM evaluation, False otherwise
    """
    workflow = log_data.get("workflow", {})
    llm_eval = workflow.get("evaluations", {}).get("llm", {})
    
    # Check if evaluation is already completed
    if llm_eval.get("completed", False):
        return False
    
    # Check if evaluation failed too many times (max 3 retries)
    if llm_eval.get("failed", False) and llm_eval.get("retry_count", 0) >= 3:
        return False
    
    return True


def mark_evaluation_complete(log_data: dict, eval_type: str, model: str) -> dict:
    """
    Mark evaluation as completed in workflow state.
    
    Args:
        log_data: Parsed log JSON data
        eval_type: Type of evaluation ("llm", "automated")
        model: Model used for evaluation
    
    Returns:
        Updated log data
    """
    if "workflow" not in log_data:
        log_data["workflow"] = {"evaluations": {}}
    
    if "evaluations" not in log_data["workflow"]:
        log_data["workflow"]["evaluations"] = {}
    
    log_data["workflow"]["evaluations"][eval_type] = {
        "completed": True,
        "date": datetime.now().isoformat(),
        "model": model,
        "failed": False,
        "error": None
    }
    
    return log_data


def mark_evaluation_failed(log_data: dict, eval_type: str, error: str) -> dict:
    """
    Mark evaluation as failed in workflow state.
    
    Args:
        log_data: Parsed log JSON data
        eval_type: Type of evaluation ("llm", "automated")  
        error: Error message
    
    Returns:
        Updated log data
    """
    if "workflow" not in log_data:
        log_data["workflow"] = {"evaluations": {}}
    
    if "evaluations" not in log_data["workflow"]:
        log_data["workflow"]["evaluations"] = {}
    
    current_eval = log_data["workflow"]["evaluations"].get(eval_type, {})
    retry_count = current_eval.get("retry_count", 0) + 1
    
    log_data["workflow"]["evaluations"][eval_type] = {
        "completed": False,
        "failed": True,
        "error": error,
        "retry_count": retry_count,
        "last_attempt": datetime.now().isoformat()
    }
    
    return log_data


def log_session_from_dict(log_filepath: str, session_data: dict):
    """
    Try to construct a SessionLog from raw dict.
    Fallback to raw JSON with error flag if schema validation fails.
    """
    try:
        session_log = SessionLog(**session_data)
        log_session(log_filepath, session_log)
    except ValidationError as e:
        print(f"[⚠️ Schema Validation] {e}")
        session_data["_SCHEMA_VALIDATION_ERROR"] = str(e)
        json_string = json.dumps(session_data, indent=2)
        Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(log_filepath, "w", encoding="utf-8") as f:
            f.write(json_string)
