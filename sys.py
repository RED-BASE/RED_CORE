import pandas as pd

# Load flat logs (edit path if needed)
df = pd.read_csv("logs/flattened/flat_logs.csv")

# The four system prompts to compare
sys_prompts = [
    'sys_none_00:latest',
    'sys_guardrail_decay_01:latest',
    'sys_guardrail_decay_02:latest',
    'sys_guardrail_decay_03:latest'
]

print("\n=== AVG DRIFT BY TURN | SYSTEM PROMPT | MODEL ===")
for sys_prompt in sys_prompts:
    print(f"\n--- {sys_prompt} ---")
    sub = df[df['system_prompt_tag'] == sys_prompt]
    if sub.empty:
        print("No data for this prompt.")
        continue
    pivot = sub.pivot_table(
        index='turn_index',
        columns='model',
        values='drift_score',
        aggfunc='mean'
    )
    print(pivot.round(3))
    # Uncomment below to save as CSV for viz or deeper analysis
    # pivot.to_csv(f"drift_by_turn_{sys_prompt.replace(':','_')}.csv")

print("\n=== Done. Each table: rows=turns, cols=models, values=avg drift ===")
