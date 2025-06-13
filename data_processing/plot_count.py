import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Load CSV summary
csv_path = "/data2/ac2220/data_handling/inflammatory_threshold_summary.csv"
df = pd.read_csv(csv_path)

# Drop rows with missing TI
df = df.dropna(subset=["TI"])

# Define thresholds and colors
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
colors = ['blue', 'green', 'orange', 'purple', 'red']

plt.figure(figsize=(12, 8))

# For each threshold, fit regression and plot
for i, t in enumerate(thresholds):
    y_col = f"inflamm_norm_{t}"
    if y_col not in df.columns:
        continue

    df_subset = df.dropna(subset=[y_col])
    X = df_subset['TI'].values.reshape(-1, 1)
    y = df_subset[y_col].values

    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # Compute correlation
    r, p = pearsonr(df_subset['TI'], y)
    print(f"Threshold {t:.1f} → r = {r:.3f}, p = {p:.3g}")

    # Plot data points and regression line
    plt.scatter(X, y, alpha=0.4, label=f"Data (≥{t})", color=colors[i], edgecolors='k', s=25)
    plt.plot(X, y_pred, linewidth=2, color=colors[i],
             label=f"Fit (≥{t})\n$r$ = {r:.2f}, $p$ = {p:.1g}")

plt.xlabel("TI Grade", fontsize=13)
plt.ylabel("Normalised Inflammatory Count", fontsize=13)
plt.title("Normalised Inflammatory Cell Count vs. TI Grade\nacross Probability Thresholds", fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("inflammatory_vs_TI_all_thresholds.png", dpi=300)
plt.close()
print("✅ Plot saved as 'inflammatory_vs_TI_all_thresholds.png'")
