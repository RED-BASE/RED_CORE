import pandas as pd

df = pd.read_csv("logs/flattened/flat_logs.csv")  # Or your CSV path

# 1. Find all prompt headers that contain 'alignment' or 'reflect' and have nonzero drift
reflective = df[
    (df['prompt_header'].str.contains('align', case=False) | df['prompt_header'].str.contains('reflect', case=False))
    & (df['drift_score'] > 0)
]

print("\n=== Top Reflective/Alignment Prompt Headers by Average Drift ===")
print(reflective.groupby('prompt_header')['drift_score'].mean().sort_values(ascending=False).head(10))

# 2. For each such header, print the max drift per run (to check for escalation)
print("\n=== Max Drift by Run for Top Reflection Prompts ===")
for header in reflective['prompt_header'].unique():
    runs = reflective[reflective['prompt_header'] == header]
    by_run = runs.groupby('isbn_run_id')['drift_score'].max().sort_values(ascending=False)
    print(f"\nPrompt Header: {header}")
    print(by_run.head(3))

# 3. Extra: Print a sample of "escalating" runs (drift increases turn-over-turn)
print("\n=== Example Escalating Reflection Runs ===")
for run_id, run in reflective.groupby('isbn_run_id'):
    scores = run.sort_values('turn_index')['drift_score'].tolist()
    if len(scores) >= 2 and scores[-1] > scores[0]:
        print(f"Run: {run_id} | Drift: {scores}")
