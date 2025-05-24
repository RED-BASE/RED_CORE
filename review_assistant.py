from pathlib import Path
import json

def display_turn(turn):
    print(f"\n--- Turn {turn.get('turn_index', '?')} ---")
    print(f"[Prompt]\n{turn.get('rendered_prompt')}")
    print(f"\n[Model Output]\n{turn.get('model_output')}")
    print(f"\n[Drift Score]: {turn.get('drift_score', 'N/A')}")
    print(f"[Drift Notes]: {turn.get('drift_notes', [])}")
    print(f"[Review Status]: {turn.get('review_status', 'N/A')}")
    print(f"[Human Reviewed]: {turn.get('human_reviewed', False)}")

def review_log_file(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
    for turn in data.get("turns", []):
        if turn.get("human_reviewed"):
            continue

        display_turn(turn)

        # Add optional review note
        note = input("📝 Add review note (or leave blank): ").strip()
        if note:
            turn["review_note"] = note

        response = input("✅ Mark this turn as reviewed? [y/N]: ").strip().lower()
        if response == 'y':
            turn["human_reviewed"] = True
            updated = True

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Updated {path.name} with human_reviewed: true")
    else:
        print(f"⏭ No changes made to {path.name}")

    return updated

# 🔒 Targeting logs/raw/
log_dir = Path("logs/raw")
log_files = list(log_dir.glob("*.json"))
processed = 0

for path in log_files:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            continue
    if all(t.get("human_reviewed") for t in data.get("turns", [])):
        continue
    review_log_file(path)
    processed += 1

print(f"\n🎯 Review complete. {processed} log(s) processed.")
