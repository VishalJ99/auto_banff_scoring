import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load your summary CSV
csv_path = "data_handling/inflammatory_threshold_summary.csv"
df = pd.read_csv(csv_path)

# Drop rows with missing values
df = df.dropna(subset=["TI", "cortex_patch_ratio"])

# Compute Pearson correlation
r_ratio, p_ratio = pearsonr(df["TI"], df["cortex_patch_ratio"])

print(f"TI vs cortex_patch_ratio → r = {r_ratio:.3f}, p = {p_ratio:.4g}")

# Plotting
plt.figure(figsize=(6, 5))
plt.scatter(df["TI"], df["cortex_patch_ratio"], alpha=0.6, edgecolors='k')
plt.title(f"Cortex Patch Ratio vs TI (r={r_ratio:.2f}, p={p_ratio:.2g})")
plt.xlabel("TI Score")
plt.ylabel("Cortex Patch Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig("data_handling/ti_vs_cortex_patch_ratio.png", dpi=300)
print("📊 Saved plot to: data_handling/ti_vs_cortex_patch_ratio.png")