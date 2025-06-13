import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from pathlib import Path

# === Update this path if needed ===
csv_path = "/data2/ac2220/data_handling/cluster_metrics_eps37.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Create output directory
output_dir = Path("data_handling")
output_dir.mkdir(parents=True, exist_ok=True)

# Drop missing TI values
df = df.dropna(subset=["TI"])

# Define metrics to evaluate
metrics = [
    "num_clusters",
    "clusters_per_patch",
    "mean_cluster_size",
    "max_cluster_size",
    "median_cluster_size",
    "cells_in_clusters",
    "fraction_clustered_cells",
    "large_cluster_count",
    "fraction_large_clusters",
    "wbc_clustering_index"
]

# Evaluate each metric
for metric in metrics:
    if metric not in df.columns:
        print(f"⚠️ Skipping missing column: {metric}")
        continue

    X = df["TI"].values
    y = df[metric].values

    # Pearson correlation
    r, p = pearsonr(X, y)
    print(f"{metric}: r = {r:.3f}, p = {p:.3g}")

    # Linear regression fit
    reg = LinearRegression()
    reg.fit(X.reshape(-1, 1), y)
    y_pred = reg.predict(X.reshape(-1, 1))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, edgecolors='k', alpha=0.6, label='Data')
    plt.plot(X, y_pred, color='red', label=f'Linear fit (r={r:.2f}, p={p:.2g})')
    plt.xlabel("TI score")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.title(f"{metric.replace('_', ' ').capitalize()} vs TI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    filename = output_dir / f"{metric}_vs_TI_eps37.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {filename}")
