# log_utils.py

from app.core.log_schema import SessionLog
from pydantic import ValidationError
import json
from pathlib import Path

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
