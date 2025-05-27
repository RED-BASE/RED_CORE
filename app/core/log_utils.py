"""
Utility helpers for unified JSONL logging, run ID generation, and log hygiene.
"""

import json
import os
import datetime
import hashlib
import time
import gzip
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

from pydantic import ValidationError
from utils.log_schema import UnifiedEvent

# --- REGISTRY IMPORTS ---
try:
    from config import LOG_DIR, get_model_code
except ImportError:
    LOG_DIR = "logs"
    def get_model_code(model_name): return "UNK"

# ---------------------------------------------------------------------
# Unified JSONL Logging
# ---------------------------------------------------------------------
def log_unified_event(log_filepath: str, event_data: Dict[str, Any]):
    """Append one validated event to a JSONL log file."""
    try:
        event_data["evaluator_version"] = event_data.get("evaluator_version", "unknown")
        log_entry = UnifiedEvent(**event_data)
        json_string = log_entry.model_dump_json()
    except ValidationError as e:
        print(f"[⚠️ Schema] Validation error: {e}")
        event_data["_SCHEMA_VALIDATION_ERROR"] = str(e)
        json_string = json.dumps(event_data, default=str)
    except Exception as unexpected:
        print(f"[❌] Unexpected logging error: {unexpected}")
        return

    Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(json_string + "\n")

# ---------------------------------------------------------------------
# Run-ID & Hash Utilities
# ---------------------------------------------------------------------
def create_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision

def create_config_hash(config_dict: Dict[str, Any]) -> str:
    stable = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:10]

# ---------------------------------------------------------------------
# Log File Hygiene
# ---------------------------------------------------------------------
def compress_and_rotate(log_path: str, keep_days: int = 30):
    """Gzip a .jsonl log and delete files older than N days."""
    log_file = Path(log_path)
    compressed = log_file.with_suffix(log_file.suffix + ".gz")

    if log_file.exists():
        with open(log_file, "rb") as src, gzip.open(compressed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        log_file.unlink(missing_ok=True)

    cutoff = time.time() - keep_days * 86400
    for gz in log_file.parent.glob("*.jsonl.gz"):
        if gz.stat().st_mtime < cutoff:
            gz.unlink(missing_ok=True)

# ---------------------------------------------------------------------
# Readable Run ID Generator
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Output Path Helper
# ---------------------------------------------------------------------
def get_log_path(run_id: str, subfolder: str = "raw") -> str:
    return str(Path(LOG_DIR) / subfolder / f"{run_id}.json")
