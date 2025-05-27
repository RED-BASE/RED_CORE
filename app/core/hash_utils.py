import json
import hashlib
import os

def embed_log_hash(json_path: str, write_to_new_file: bool = False, output_dir: str = None) -> str:
    """
    Reads a JSON log file, computes its SHA-256 hash (excluding any existing hash field),
    embeds the hash under the 'hash' key.

    By default, overwrites the original file. If `write_to_new_file` is True,
    writes to `<original_name>.hash.json` or to `output_dir/<original_name>.hash.json`.

    Returns the computed hash string.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    data.pop("hash", None)

    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    hash_str = hashlib.sha256(raw).hexdigest()
    data["hash"] = hash_str

    if write_to_new_file:
        base_name = os.path.basename(json_path)
        name_root = base_name.rsplit(".", 1)[0]  # strip .json
        new_name = f"{name_root}.hash.json"
        save_path = os.path.join(output_dir, new_name) if output_dir else os.path.join(os.path.dirname(json_path), new_name)
    else:
        save_path = json_path

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    return hash_str


def verify_log_hash(json_path: str) -> bool:
    """
    Verifies that the embedded 'hash' field in a JSON log file matches the actual content.
    Returns True if the hash matches, False otherwise.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    embedded_hash = data.pop("hash", None)
    if not embedded_hash:
        raise ValueError("No 'hash' field found in the JSON file.")

    raw = json.dumps(data, sort_keys=True).encode("utf-8")
    computed_hash = hashlib.sha256(raw).hexdigest()

    return embedded_hash == computed_hash

def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
