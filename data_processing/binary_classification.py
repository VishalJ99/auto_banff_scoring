import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,  # Added this import
)

# === UPDATE THIS PATH ===
csv_path = "/data2/ac2220/data_handling/cluster_metrics_with_zscores.csv"

# Load and clean data
df = pd.read_csv(csv_path)
df = df.dropna(subset=["TI", "optimized_inflammation_score"])

# Ground-truth binary labels: high TI if TI >= 2
df["high_TI"] = (df["TI"] >= 2).astype(int)
y_true = df["high_TI"].values
y_score = df["optimized_inflammation_score"].values

# Store metrics
thresholds = np.linspace(y_score.min(), y_score.max(), 200)
metrics = []

for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)  # Added kappa calculation
    youden_j = sensitivity + specificity - 1
    metrics.append((t, sensitivity, specificity, accuracy, kappa, youden_j))  # Added kappa to tuple

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics, columns=["threshold", "sensitivity", "specificity", "accuracy", "kappa", "youden_j"])  # Added kappa column

# Find threshold that maximizes Youden's J
best_row = metrics_df.loc[metrics_df["youden_j"].idxmax()]
best_threshold = best_row["threshold"]
print(f"\n✅ Optimal threshold by Youden's J: {best_threshold:.3f}")
print(f"Sensitivity: {best_row['sensitivity']:.3f}")
print(f"Specificity: {best_row['specificity']:.3f}")
print(f"Accuracy: {best_row['accuracy']:.3f}")
print(f"Cohen's Kappa: {best_row['kappa']:.3f}")  # Added kappa output
print(f"Youden's J: {best_row['youden_j']:.3f}")

# ROC AUC (not threshold-dependent)
auc = roc_auc_score(y_true, y_score)
print(f"AUC: {auc:.3f}")

# === BONUS: Fixed threshold (0) analysis ===
print(f"\n📊 Fixed threshold = 0 analysis:")
y_pred_fixed = (y_score >= 0).astype(int)
fixed_sensitivity = recall_score(y_true, y_pred_fixed)
tn_fixed, fp_fixed, fn_fixed, tp_fixed = confusion_matrix(y_true, y_pred_fixed).ravel()
fixed_specificity = tn_fixed / (tn_fixed + fp_fixed)
fixed_accuracy = accuracy_score(y_true, y_pred_fixed)
fixed_kappa = cohen_kappa_score(y_true, y_pred_fixed)

print(f"Sensitivity: {fixed_sensitivity:.3f}")
print(f"Specificity: {fixed_specificity:.3f}")
print(f"Accuracy: {fixed_accuracy:.3f}")
print(f"Cohen's Kappa: {fixed_kappa:.3f}")

# Optional: plot metrics vs threshold
plt.figure(figsize=(12, 6))
plt.plot(metrics_df["threshold"], metrics_df["sensitivity"], label="Sensitivity")
plt.plot(metrics_df["threshold"], metrics_df["specificity"], label="Specificity")
plt.plot(metrics_df["threshold"], metrics_df["accuracy"], label="Accuracy")
plt.plot(metrics_df["threshold"], metrics_df["kappa"], label="Cohen's Kappa")  # Added kappa to plot
plt.plot(metrics_df["threshold"], metrics_df["youden_j"], label="Youden's J", linestyle='--')
plt.axvline(best_threshold, color="grey", linestyle=":", label=f"Best threshold = {best_threshold:.2f}")
plt.axvline(0, color="red", linestyle=":", alpha=0.7, label="Fixed threshold = 0")  # Added fixed threshold line
plt.xlabel("Threshold on inflammation score")
plt.ylabel("Metric value")
plt.title("Binary classification metrics vs threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_metrics_plot.png", dpi=300)
print("📈 Plot saved to threshold_metrics_plot.png")

# === BONUS: Kappa interpretation ===
def interpret_kappa(kappa_value):
    if kappa_value < 0:
        return "Poor (worse than chance)"
    elif kappa_value < 0.20:
        return "Slight"
    elif kappa_value < 0.40:
        return "Fair"
    elif kappa_value < 0.60:
        return "Moderate"
    elif kappa_value < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"

print(f"\n📋 Kappa Interpretation:")
print(f"Fixed threshold (0): κ = {fixed_kappa:.3f} ({interpret_kappa(fixed_kappa)})")
print(f"Optimal threshold: κ = {best_row['kappa']:.3f} ({interpret_kappa(best_row['kappa'])})")
print(f"\n🔬 Literature comparison:")
print(f"Your method: κ = {fixed_kappa:.3f}")
print(f"Banff inter-observer (literature): κ = 0.40 (Fair-Moderate)")