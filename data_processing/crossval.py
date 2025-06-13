import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# Load your data
csv_path = "data_handling/cluster_metrics_eps37.csv"  # Update path if needed
df = pd.read_csv(csv_path)

# Drop rows with missing values
df = df.dropna(subset=["TI", "inflammatory_per_cortex_patch", "wbc_clustering_index"])

# Create z-score columns
df['z_norm'] = (df['inflammatory_per_cortex_patch'] - df['inflammatory_per_cortex_patch'].mean()) / df['inflammatory_per_cortex_patch'].std()
df['z_wci'] = (df['wbc_clustering_index'] - df['wbc_clustering_index'].mean()) / df['wbc_clustering_index'].std()

# Define alpha/beta sweep
steps = np.arange(0.0, 1, 0.01)
weight_pairs = [(round(a, 2), round(1 - a, 2)) for a in steps]

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for alpha, beta in weight_pairs:
    fold_corrs = []
    for train_idx, test_idx in kf.split(df):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

        # Apply z-score transformation using train stats
        z_norm_mean = df_train['inflammatory_per_cortex_patch'].mean()
        z_norm_std = df_train['inflammatory_per_cortex_patch'].std()
        z_wci_mean = df_train['wbc_clustering_index'].mean()
        z_wci_std = df_train['wbc_clustering_index'].std()

        z_norm_test = (df_test['inflammatory_per_cortex_patch'] - z_norm_mean) / z_norm_std
        z_wci_test = (df_test['wbc_clustering_index'] - z_wci_mean) / z_wci_std

        score = alpha * z_norm_test + beta * z_wci_test
        ti = df_test['TI'].values

        if len(ti) > 1:
            r, _ = pearsonr(score, ti)
            fold_corrs.append(r)

    if fold_corrs:
        avg_r = np.mean(fold_corrs)
        results.append((alpha, beta, round(avg_r, 4)))

# Print results
results = sorted(results, key=lambda x: -x[2])
print("Top performing alpha/beta pairs (sorted by average Pearson r):")
for alpha, beta, avg_r in results:
    print(f"α = {alpha:.2f}, β = {beta:.2f} → Avg r = {avg_r:.4f}")