import pandas as pd

# EDIT path if your logs are elsewhere
df = pd.read_csv("logs/flattened/flat_logs.csv")

print("\n=== DRIFT ANALYSIS BY SYSTEM PROMPT ===")

# Aggregate stats for each system_prompt_tag
drift_stats = df.groupby('system_prompt_tag').agg(
    total_turns=('turn_index', 'count'),
    drift_turns=('drift_score', lambda x: (x > 0).sum()),
    avg_drift=('drift_score', 'mean'),
    max_drift=('drift_score', 'max')
).sort_values(by='avg_drift', ascending=False)

print("\n--- Top 15 System Prompts by Average Drift ---")
print(drift_stats.head(15))

# Optional: print all system prompts
# print(drift_stats)

# Deep-dive: system prompt + model interactions
print("\n--- Top 15 System Prompt + Model Combinations by Avg Drift ---")
prompt_model = df.groupby(['system_prompt_tag', 'model']).agg(
    avg_drift=('drift_score', 'mean'),
    max_drift=('drift_score', 'max'),
    drift_turns=('drift_score', lambda x: (x > 0).sum())
).sort_values(by='avg_drift', ascending=False)

print(prompt_model.head(15))

print("\n--- Done. Tweak, analyze, or export as needed. ---")

# Optionally, write to CSV for external analysis (uncomment as needed):
# drift_stats.to_csv("drift_by_system_prompt.csv")
# prompt_model.to_csv("drift_by_sys_prompt_and_model.csv")

# Per-model drift for just sys_guardrail_decay_01:latest
o1_drift = df[df['system_prompt_tag'] == 'sys_guardrail_decay_01:latest'].groupby('model').agg(
    total_turns=('turn_index', 'count'),
    drift_turns=('drift_score', lambda x: (x > 0).sum()),
    avg_drift=('drift_score', 'mean'),
    max_drift=('drift_score', 'max')
).sort_values(by='avg_drift', ascending=False)

print("\n--- Per-Model Drift for sys_guardrail_decay_01:latest ---")
print(o1_drift)
