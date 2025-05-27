import os
import csv
from datetime import datetime
import subprocess

# Correct location
FOLDER = os.path.join("meta", "friction_audit_logs")
CSV_FILE = os.path.join(FOLDER, "friction_audit_log.csv")
SCRIPT_VERSION = os.path.basename(__file__)

QUESTIONS = [
    ("Actionability (1–5):", "actionability"),
    ("Novelty (1–5):", "novelty"),
    ("Did you feel *less* willing to use the tool this session? (y/n):", "avoidance"),
    ("Did you share with a third party? (y/n):", "third_party"),
    ("Did the session surface any real contradictions/discomfort? (y/n):", "contradiction"),
    ("Describe in a sentence:", "contradiction_note"),
    ("Did you doubt your problem (1), or the tool/process itself (2), or neither (0)?", "self_doubt"),
    ("Any other notes?", "notes"),
]

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "NO_GIT_HASH"

def ensure_folder():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

def write_header_if_needed():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            header = ["timestamp", "git_commit", "script"] + [q[1] for q in QUESTIONS]
            writer.writerow(header)

def log_session():
    print("Friction Audit Log - The Well-Digger")
    answers = []
    for prompt, _ in QUESTIONS:
        ans = input(prompt + " ").strip()
        answers.append(ans)
    now = datetime.now().isoformat()
    git_hash = get_git_hash()
    with open(CSV_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, git_hash, SCRIPT_VERSION] + answers)
    print(f"Session logged to {CSV_FILE}")
