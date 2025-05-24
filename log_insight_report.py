import pandas as pd

df = pd.read_csv("flat_logs.csv")

print("\n==== BASIC STATS ====")
print(f"Total rows: {len(df)}")
print("Unique runs:", df['isbn_run_id'].nunique())
print("Models:", df['model'].unique())
print("Nonzero drift turns:", (df['drift_score'] > 0).sum())
print("Missing model_output:", df['model_output'].isnull().sum())

# ---- 1. Model with the Most High-Drift Events ----
print("\n==== WHICH MODEL SPIKES THE MOST? ====")
spikes = df[df['drift_score'] > 0.3].groupby('model').size().sort_values(ascending=False)
print(spikes)

# ---- 2. Most "Boundary Reflection" Per Run ----
print("\n==== RUNS WITH THE MOST BOUNDARY REFLECTIONS ====")
df['reflection_count'] = df['drift_notes'].apply(lambda x: str(x).count('boundary_reflection'))
top_reflect = df.groupby('isbn_run_id')['reflection_count'].sum().sort_values(ascending=False).head(5)
print(top_reflect)

# ---- 3. Run(s) With the Longest Consecutive Drift Streak ----
print("\n==== RUNS WITH THE LONGEST DRIFT STREAKS ====")
max_streaks = {}
for run_id, run in df.groupby('isbn_run_id'):
    streak, max_streak = 0, 0
    for val in run['drift_score']:
        if val > 0:
            streak += 1
        else:
            max_streak = max(max_streak, streak)
            streak = 0
    max_streak = max(max_streak, streak)
    if max_streak > 1:
        max_streaks[run_id] = max_streak
if max_streaks:
    top = sorted(max_streaks.items(), key=lambda x: -x[1])[:5]
    print("Run ID | Max consecutive drift turns")
    for rid, streak in top:
        print(rid, "|", streak)
else:
    print("No consecutive drift streaks found.")

# ---- 4. Turns Where Model Output is Missing Despite High Drift ----
print("\n==== MISSING OUTPUT WHERE DRIFT WAS HIGH ====")
anomalies = df[(df['model_output'].isnull()) & (df['drift_score'] > 0.1)]
print(anomalies[['isbn_run_id','turn_index','drift_score','drift_notes']])

# ---- 5. Outlier: Highest Drift Note Diversity in a Run ----
print("\n==== RUNS WITH THE MOST UNIQUE DRIFT NOTES ====")
def parse_notes(x):
    try:
        return set(eval(x)) if isinstance(x, str) and x.startswith('[') else set()
    except Exception:
        return set()
note_counts = df.groupby('isbn_run_id')['drift_notes'].apply(lambda notes: len(set.union(*(parse_notes(n) for n in notes))))
print(note_counts.sort_values(ascending=False).head(5))

print("\n==== BONUS: WHICH PROMPT HEADERS TRIGGER MOST DRIFT? ====")
header_drift = df[df['drift_score'] > 0.1].groupby('prompt_header').size().sort_values(ascending=False)
print(header_drift.head(5))

print("\n==== DONE. Save or tweak this script for deeper dives. ====")
