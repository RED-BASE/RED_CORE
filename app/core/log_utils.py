# log_utils.py

from app.core.log_schema import SessionLog
from pydantic import ValidationError
import json
from pathlib import Path
import datetime
import hashlib
from app.config import get_model_code

def generate_readable_run_id(
    model_name: str,
    user_prompt_tag: str,
    system_prompt_tag: Optional[str] = None,
    persona: Optional[str] = None,
    experiment_code: str = "GRD"
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

    s_tag = normalize_tag(system_prompt_tag, "sys")
    u_tag = normalize_tag(user_prompt_tag, "usr")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    unique = hashlib.sha1(f"{model_name}{system_prompt_tag}{user_prompt_tag}{persona}{timestamp}".encode()).hexdigest()[:6].upper()

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
