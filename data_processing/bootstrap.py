import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("data_handling/cluster_metrics_with_zscores.csv")  # Make sure this has 'TI' and 'optimized_inflammation_score'

# Drop any rows with missing values
df = df.dropna(subset=["TI", "optimized_inflammation_score"])

# Set up variables
TI = df["TI"].values
score = df["optimized_inflammation_score"].values
n = len(df)

# Bootstrap settings
n_iterations = 1000
rng = np.random.default_rng(seed=42)  # Reproducible results
r_values = []

for _ in range(n_iterations):
    indices = rng.integers(0, n, n)  # Sample with replacement
    sample_TI = TI[indices]
    sample_score = score[indices]
    r, _ = pearsonr(sample_TI, sample_score)
    r_values.append(r)

# Compute confidence interval
ci_lower = np.percentile(r_values, 2.5)
ci_upper = np.percentile(r_values, 97.5)
mean_r = np.mean(r_values)

# Output results
print(f"Bootstrap Pearson r = {mean_r:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Optional: Plot distribution
plt.hist(r_values, bins=40, edgecolor='k', alpha=0.7)
plt.axvline(ci_lower, color='red', linestyle='--', label=f"2.5% = {ci_lower:.3f}")
plt.axvline(ci_upper, color='red', linestyle='--', label=f"97.5% = {ci_upper:.3f}")
plt.axvline(mean_r, color='blue', linestyle='-', label=f"Mean r = {mean_r:.3f}")
plt.title("Bootstrap Distribution of Pearson Correlation (r)")
plt.xlabel("r")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("data_handling/bootstrap_r_distribution.png", dpi=300)