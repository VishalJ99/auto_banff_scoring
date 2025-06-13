import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Load your data ===
csv_path = "/data2/ac2220/data_handling/cluster_metrics_with_zscores.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["TI", "optimized_inflammation_score"])

# === Ground truth and predictions ===
df["high_TI"] = (df["TI"] >= 2).astype(int)
df["predicted_high"] = (df["optimized_inflammation_score"] >= 0).astype(int)  # Threshold = 0

y_true = df["high_TI"]
y_pred = df["predicted_high"]

# === Compute confusion matrix ===
cm = confusion_matrix(y_true, y_pred)
labels = ["TI < 2", "TI ≥ 2"]

# === Plot ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Threshold = 0)")
plt.savefig("confusion_matrix.png", dpi=300)
plt.tight_layout()
