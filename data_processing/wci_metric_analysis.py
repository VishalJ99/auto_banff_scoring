import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load your dataframe
csv_path = "data_handling/cluster_metrics_eps37.csv"  # Update if needed
df = pd.read_csv(csv_path)

# Ensure columns exist
assert all(col in df.columns for col in ['TI', 'inflammatory_per_cortex_patch', 'wbc_clustering_index'])

# Drop rows with missing data
df = df.dropna(subset=['TI', 'inflammatory_per_cortex_patch', 'wbc_clustering_index'])

# Z-score features
df['z_norm'] = (df['inflammatory_per_cortex_patch'] - df['inflammatory_per_cortex_patch'].mean()) / df['inflammatory_per_cortex_patch'].std()
df['z_wci'] = (df['wbc_clustering_index'] - df['wbc_clustering_index'].mean()) / df['wbc_clustering_index'].std()

# Grid search over alpha (weight for z_norm), beta = 1 - alpha
alphas = np.linspace(0, 1, 101)
results = []

for alpha in alphas:
    beta = 1 - alpha
    df['combined_weighted'] = alpha * df['z_norm'] + beta * df['z_wci']
    r, p = pearsonr(df['TI'], df['combined_weighted'])
    results.append((alpha, r, p))

# Find best alpha
best = max(results, key=lambda x: x[1])
best_alpha, best_r, best_p = best
print(f"✅ Best correlation: r = {best_r:.3f}, p = {best_p:.2e}")
print(f"Optimal weights: alpha (z_norm) = {best_alpha:.2f}, beta (z_wci) = {1 - best_alpha:.2f}")

# Optional: plot
alphas, rs, ps = zip(*results)
plt.figure(figsize=(8, 5))
plt.plot(alphas, rs, marker='o')
plt.xlabel("Weight on Z-normalized inflammatory density (α)")
plt.ylabel("Pearson r with TI score")
plt.title("Optimizing weight between z_norm and z_wci")
plt.grid(True)
plt.tight_layout()
plt.savefig("data_handling/optimized_z_weight_correlation.png", dpi=300)
