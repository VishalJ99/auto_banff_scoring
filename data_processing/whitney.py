import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Load the CSV file
df = pd.read_csv("inflammatory_threshold_summary.csv")

# Choose the thresholded score you want to evaluate
score_column = "inflamm_norm_0.5"  # ← change this if needed

# Group by TI score
print("TI Grade | n slides | Mean Score | Std Dev | Range")
ti_grades = sorted(df['TI'].dropna().unique())

scores_by_ti = {}
for ti in ti_grades:
    scores = df[df['TI'] == ti][score_column].dropna().values
    scores_by_ti[int(ti)] = scores  # store for Mann-Whitney later
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    r_min, r_max = np.min(scores), np.max(scores)
    print(f"TI-{int(ti)}     | {len(scores):8} | {mean:.4f}      | ±{std:.4f}  | [{r_min:.4f}, {r_max:.4f}]")

# Mann-Whitney U tests between adjacent grades
print("\nMann-Whitney U test p-values between adjacent TI grades:")
for g1, g2 in zip(ti_grades[:-1], ti_grades[1:]):
    u_stat, p_val = mannwhitneyu(scores_by_ti[int(g1)], scores_by_ti[int(g2)], alternative='two-sided')
    print(f"TI-{int(g1)} vs TI-{int(g2)}: p = {p_val:.4g}")
