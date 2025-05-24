import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd

# 📁 Ensure project root is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.metrics import score_session_log  # Adjusted for /metrics/metrics.py

def load_scored_log(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame([
        {
            "turn_index": turn.get("turn_index", i),
            "agent_name": turn.get("agent_name", data.get("model", "unknown")),
            "variant_id": turn.get("variant_id", f"v{i}"),
            **turn.get("metrics", {})
        }
        for i, turn in enumerate(data["turns"])
    ])

from unused.drift_taxonomy import label_drift

def diff_scores(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    joined = df1.set_index("turn_index").join(
        df2.set_index("turn_index"), lsuffix="_a", rsuffix="_b"
    ).reset_index()

    diffs = []
    for _, row in joined.iterrows():
        row_diff = {
            "turn_index": row["turn_index"],
            "variant_id": row.get("variant_id_a", "unknown"),
            "agent_name_a": row.get("agent_name_a", "A"),
            "agent_name_b": row.get("agent_name_b", "B")
        }
        for metric in ["refusal", "volition", "consent", "persistence", "drift"]:
            a, b = row.get(f"{metric}_a", 0), row.get(f"{metric}_b", 0)
            delta = b - a
            if delta != 0:
                row_diff[metric] = f"{a} → {b} ({'+' if delta > 0 else ''}{delta})"
        row_diff['drift_label'] = label_drift(row_diff)
        diffs.append(row_diff)

    for row in diffs:
        row['drift_label'] = label_drift(row)

    return pd.DataFrame(diffs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diff two scored .json logs.")
    parser.add_argument("--log_a", required=True, help="Path to first log (baseline)")
    parser.add_argument("--log_b", required=True, help="Path to second log (comparison)")
    args = parser.parse_args()

    df_a = load_scored_log(args.log_a)
    df_b = load_scored_log(args.log_b)
    df_diff = diff_scores(df_a, df_b)

    print("\n=== Metric Deltas by Turn ===")
    print(df_diff.to_string(index=False))
