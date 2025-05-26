#!/usr/bin/env python3
import sys
import re
import os

# Patterns to block (write/mutation operations)
WRITE_PATTERNS = [
    r'open\s*\(.*[\"\"][wax][\"\"]',  # open( with 'w', 'a', or 'x'
    r'\.to_csv\s*\(',
    r'\.to_json\s*\(',
    r'\.write_json\s*\(',
    r'os\.rename\s*\(',
    r'os\.remove\s*\(',
    r'shutil\.move\s*\(',
    r'shutil\.copy\s*\(',
]

# Whitelisted scripts (full relative paths)
WHITELIST = {
    'app/cli/run_experiments.py',
    'meta/friction_audit.py',
    'dev_log/reflections/create_log_entry.sh',
}

# Helper to check if a file is whitelisted
def is_whitelisted(filepath):
    return filepath in WHITELIST

def main():
    # Get staged files from pre-commit (all args after script name)
    files = [f for f in sys.argv[1:] if os.path.isfile(f)]
    blocked = False
    for filepath in files:
        relpath = os.path.relpath(filepath)
        if is_whitelisted(relpath):
            continue  # Whitelisted file
        # For all other files, scan for forbidden patterns
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for pat in WRITE_PATTERNS:
                    if re.search(pat, line):
                        print(f"Commit blocked: File write operation detected in {relpath} (line {i}). Only allowed in orchestrator and meta-tools. See CONTRIBUTING.md for allowed patterns.")
                        blocked = True
    if blocked:
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main() 