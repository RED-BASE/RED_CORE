"""
flattener.py

Flattens all run-level JSON logs into a turn-level CSV for downstream analytics.

Usage:
    python flattener.py --log-dir logs/raw --output flattened_guardrail_logs.csv
"""
import json
import pandas as pd
from pathlib import Path
import argparse

def flatten_guardrail_logs_from_dir(log_dir: str) -> pd.DataFrame:
    """Flatten all valid guardrail logs from a directory into a single DataFrame."""
    flattened_data = []
    log_dir = Path(log_dir)

    for log_file in log_dir.glob("*.json"):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)

            base_info = {
                "isbn_run_id": log.get("isbn_run_id"),
                "model": log.get("model"),
                "model_code": log.get("model_code"),
                "model_vendor": log.get("model_vendor"),
                "system_prompt_tag": log.get("system_prompt_tag"),
                "system_prompt_hash": log.get("system_prompt_hash"),
                "user_prompt_hash": log.get("user_prompt_hash"),
                "persona": log.get("persona"),
                "experiment_id": log.get("experiment_id"),
                "scenario_hash": log.get("scenario_hash")
            }

            for turn in log.get("turns", []):
                flattened_data.append({
                    **base_info,
                    "turn_index": turn.get("turn_index"),
                    "variant_id": turn.get("meta", {}).get("variant_id"),
                    "prompt_header": turn.get("meta", {}).get("prompt_header"),
                    "rendered_prompt": turn.get("rendered_prompt"),
                    "user_input": turn.get("user_input"),
                    "model_output": turn.get("model_output"),
                    "drift_score": turn.get("drift_score"),
                    "drift_notes": turn.get("drift_notes"),
                    "containment_flags": turn.get("containment_flags"),
                    "review_status": turn.get("review_status")
                })
        except Exception as e:
            print(f"[WARN] Failed to parse {log_file.name}: {e}")

    return pd.DataFrame(flattened_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten raw logs to a turn-level CSV.")
    parser.add_argument("--log-dir", default="logs/raw", help="Directory containing JSON logs")
    parser.add_argument("--output", default="logs/flattened/flat_logs.csv", help="Output CSV file")
    args = parser.parse_args()

    df = flatten_guardrail_logs_from_dir(args.log_dir)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Flattened {len(df)} log entries to {args.output}.")
