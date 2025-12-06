import sys
import re

REQUIRED_STRINGS = [
    "Experiment Name",
    "Purpose",
    "Provenance",
    "Methods",
    "Results"
]

for filename in sys.argv[1:]:
    with open(filename, "r") as f:
        content = f.read().lower()
        for required in REQUIRED_STRINGS:
            if required.lower() not in content:
                print(f"ERROR: README in {filename} missing section: {required}")
                sys.exit(1)
        if "template" in content or "replace" in content:
            print(f"ERROR: README in {filename} contains unresolved template text.")
            sys.exit(1)
print("All experiment READMEs validated.")
