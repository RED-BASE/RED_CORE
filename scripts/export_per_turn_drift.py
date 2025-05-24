from pathlib import Path
import json
import pandas as pd

log_dir = Path("logs/raw")  # Make sure this path is correct
turn_data = []

for log_path in log_dir.glob("*.json"):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON from {log_path.name} (JSONDecodeError)")
        continue
    except Exception as e:
        print(f"❌ Failed to process {log_path.name}: {e}")
        continue

    # Top-level data
    model = data.get("model")
    experiment_id = data.get("experiment_id")
    
    # Refined sys_prompt_tag extraction
    raw_sys_tag = data.get("system_prompt_tag") # Correct key name
    if raw_sys_tag and isinstance(raw_sys_tag, str) and ":" in raw_sys_tag:
        sys_tag_main = raw_sys_tag.split(":")[0]
        sys_tag_version = raw_sys_tag.split(":", 1)[1] # Capture version if present
    elif raw_sys_tag and isinstance(raw_sys_tag, str):
        sys_tag_main = raw_sys_tag
        sys_tag_version = "N/A"
    else:
        sys_tag_main = "SYS_UNKNOWN"
        sys_tag_version = "N/A"

    # Refined usr_prompt_tag extraction
    usr_tag = data.get("usr_prompt_tag")
    if not usr_tag and "exploit_path" in data and data["exploit_path"]:
        try:
            usr_tag = Path(data["exploit_path"]).stem
        except Exception: # Handle potential issues with Path object if exploit_path is malformed
            usr_tag = "USR_PATH_ERROR"
    elif not usr_tag:
        usr_tag = "USR_UNKNOWN"

    # Additional top-level fields
    mode = data.get("mode")
    temperature = data.get("temperature")
    system_prompt_hash = data.get("system_prompt_hash")
    user_prompt_hash = data.get("user_prompt_hash")
    persona_top_level = data.get("persona") # Could be top-level
    scenario_hash = data.get("scenario_hash")
    scored_value = data.get("scored") # Keep original type for now

    turns = data.get("turns", [])
    if not isinstance(turns, list): # Ensure turns is a list
        print(f"⚠️ 'turns' data in {log_path.name} is not a list. Skipping this file's turns.")
        continue

    for turn in turns:
        if not isinstance(turn, dict): # Ensure each turn is a dictionary
            print(f"⚠️ Found a non-dictionary turn item in {log_path.name}. Skipping this turn.")
            continue
            
        drift_notes_list = turn.get("drift_notes", [])
        # Ensure drift_notes_list is actually a list of strings before joining
        if isinstance(drift_notes_list, list) and all(isinstance(note, str) for note in drift_notes_list):
            drift_notes_str = "; ".join(drift_notes_list)
        elif isinstance(drift_notes_list, str): # If it's already a string (legacy or different format)
             drift_notes_str = drift_notes_list
        else:
            drift_notes_str = "" # Default for malformed drift_notes

        meta_info = turn.get("meta", {})
        variant_id = meta_info.get("variant_id") if isinstance(meta_info, dict) else None
        prompt_header = meta_info.get("prompt_header") if isinstance(meta_info, dict) else None
        
        containment_flags_list = turn.get("containment_flags", [])
        if isinstance(containment_flags_list, list) and all(isinstance(flag, str) for flag in containment_flags_list):
            containment_flags_str = "; ".join(containment_flags_list)
        else:
            containment_flags_str = ""


        turn_data.append({
            "filename": log_path.name,
            "experiment_id": experiment_id,
            "model": model,
            "sys_prompt_tag": sys_tag_main, # Store main tag
            "sys_prompt_version": sys_tag_version, # Store version separately
            "usr_prompt_tag": usr_tag,
            "mode": mode, # New
            "temperature": temperature, # New
            "system_prompt_hash": system_prompt_hash, # New
            "user_prompt_hash": user_prompt_hash, # New
            "persona": persona_top_level, # Assuming top-level persona is preferred
            "scenario_hash": scenario_hash, # New
            "scored": scored_value, # New
            "turn_index": turn.get("turn_index"),
            "drift_score": turn.get("drift_score"),
            "drift_notes": drift_notes_str,
            # "review_status": turn.get("review_status"), # Removed as requested
            # "human_reviewed": turn.get("human_reviewed", False), # Removed as requested
            "user_input": turn.get("user_input", "").strip(), # New, with fallback
            "model_output": turn.get("model_output", "").strip(),
            "rendered_prompt": turn.get("rendered_prompt", "").strip(),
            "variant_id": variant_id, # New
            "prompt_header": prompt_header, # New
            "containment_flags": containment_flags_str # New
        })

if turn_data:
    turn_df = pd.DataFrame(turn_data)
    output_path = Path("logs/summary/per_turn_drift_v2.csv") # Changed output filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    turn_df.to_csv(output_path, index=False, encoding='utf-8') # Specify encoding
    print(f"✅ Exported {len(turn_df)} rows to {output_path}")
else:
    print("ℹ️ No turn data was extracted. CSV not created.")