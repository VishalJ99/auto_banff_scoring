import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns

# Load your data
csv_path = "data_handling/cluster_metrics_eps37.csv"  # Update path if needed
df = pd.read_csv(csv_path)

# Drop missing values
df = df.dropna(subset=["TI", "inflammatory_per_cortex_patch", "wbc_clustering_index"])

# Compute z-scores
df['z_norm'] = (df['inflammatory_per_cortex_patch'] - df['inflammatory_per_cortex_patch'].mean()) / df['inflammatory_per_cortex_patch'].std()
df['z_wci'] = (df['wbc_clustering_index'] - df['wbc_clustering_index'].mean()) / df['wbc_clustering_index'].std()

# Compute optimized inflammation score
alpha = 0.56
beta = 0.44
df['optimized_inflammation_score'] = alpha * df['z_norm'] + beta * df['z_wci']

# Convert continuous score to TI grades for kappa calculation
def score_to_ti_grade(score):
    """Convert continuous inflammation score to predicted TI grade"""
    # These thresholds should be optimized based on your data
    # Using quartile-based thresholds as a starting point
    q25, q50, q75 = np.percentile(df['optimized_inflammation_score'], [25, 50, 75])
    
    if score <= q25:
        return 0
    elif score <= q50:
        return 1
    elif score <= q75:
        return 2
    else:
        return 3

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

# Logistic regression on z-scored features
X_logreg = df[['z_norm', 'z_wci']].values
y_logreg = df['TI'].astype(int).values

# Train model
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logreg.fit(X_logreg, y_logreg)

# Predict
df['logreg_pred'] = logreg.predict(X_logreg)

# Evaluate performance
print("\n🔍 Logistic Regression Classification Report:")
print(classification_report(y_logreg, df['logreg_pred']))

# Confusion matrix
cm_logreg = confusion_matrix(y_logreg, df['logreg_pred'])
kappa_logreg = cohen_kappa_score(y_logreg, df['logreg_pred'])
print(f"Cohen’s Kappa (logistic regression): {kappa_logreg:.3f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[f'Pred TI-{i}' for i in range(4)],
            yticklabels=[f'True TI-{i}' for i in range(4)])
plt.title("Logistic Regression Confusion Matrix (TI Grades)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("data_handling/logreg_ti_confusion_matrix.png", dpi=300)
print("📊 Logistic regression confusion matrix saved.")


# Alternative: Use threshold of 0 for binary classification (TI < 2 vs TI >= 2)
df['binary_true'] = (df['TI'] >= 2).astype(int)
df['binary_pred'] = (df['optimized_inflammation_score'] >= 0).astype(int)

# Calculate predicted TI grades
df['predicted_ti'] = df['optimized_inflammation_score'].apply(score_to_ti_grade)

# Save dataframe with new columns
df.to_csv("data_handling/cluster_metrics_with_zscores.csv", index=False)
print("✅ Z-scores and inflammation scores saved to: data_handling/cluster_metrics_with_zscores.csv")

# Correlation with TI
r, p = pearsonr(df['TI'], df['optimized_inflammation_score'])

# Calculate Cohen's kappa for multiclass classification
kappa_multiclass = cohen_kappa_score(df['TI'], df['predicted_ti'])

# Calculate Cohen's kappa for binary classification (TI >= 2)
kappa_binary = cohen_kappa_score(df['binary_true'], df['binary_pred'])

print(f"\n📊 Performance Metrics:")
print(f"Correlation (r): {r:.3f} (p = {p:.2e})")
print(f"Cohen's Kappa (4-class): {kappa_multiclass:.3f}")
print(f"Cohen's Kappa (binary TI≥2): {kappa_binary:.3f}")

# Create confusion matrix for multiclass
cm_multiclass = confusion_matrix(df['TI'], df['predicted_ti'])
print(f"\nConfusion Matrix (4-class prediction):")
print("Rows = True TI, Columns = Predicted TI")
print(cm_multiclass)

# Create confusion matrix for binary classification
cm_binary = confusion_matrix(df['binary_true'], df['binary_pred'])
print(f"\nConfusion Matrix (binary TI≥2 prediction):")
print("Rows = True (0=<TI2, 1=≥TI2), Columns = Predicted")
print(cm_binary)

# Fit regression line
X = df['TI'].values.reshape(-1, 1)
y = df['optimized_inflammation_score'].values
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Create figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Scatter + regression
axs[0].scatter(df['TI'], df['optimized_inflammation_score'], alpha=0.6, edgecolors='k', label="Slides")
axs[0].plot(df['TI'], y_pred, color='red', linewidth=2, label=f"Fit (r={r:.2f}, p={p:.1e})")
axs[0].set_xlabel("TI Grade")
axs[0].set_ylabel("Combined Inflammation Score")
axs[0].set_title("Scatter + Linear Fit")
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Box plot grouped by TI
df.boxplot(column='optimized_inflammation_score', by='TI', ax=axs[1], grid=False)
axs[1].set_title("Distribution by TI Grade")
axs[1].set_xlabel("TI Grade")
axs[1].set_ylabel("")

# Subplot 3: Confusion matrix heatmap
sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues', ax=axs[2],
            xticklabels=['Pred TI-0', 'Pred TI-1', 'Pred TI-2', 'Pred TI-3'],
            yticklabels=['True TI-0', 'True TI-1', 'True TI-2', 'True TI-3'])
axs[2].set_title(f'Confusion Matrix\n(κ = {kappa_multiclass:.3f})')
axs[2].set_xlabel('Predicted TI Grade')
axs[2].set_ylabel('True TI Grade')

plt.suptitle("Combined Inflammation Score Analysis", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data_handling/optimized_inflammation_analysis_with_kappa.png", dpi=300, bbox_inches='tight')
print("📈 Plot saved to: data_handling/optimized_inflammation_analysis_with_kappa.png")

# Performance by TI grade analysis
ti_grades = sorted(df['TI'].dropna().unique())
score_col = 'optimized_inflammation_score'

print(f"\n📋 Performance by TI Grade:")
print("TI Grade | n slides | Mean Score | Std Dev | Range")
scores_by_ti = {}

for ti in ti_grades:
    values = df[df['TI'] == ti][score_col].values
    scores_by_ti[int(ti)] = values
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    rmin, rmax = np.min(values), np.max(values)
    print(f"TI-{int(ti)}     | {len(values):8} | {mean:.4f}      | ±{std:.4f}  | [{rmin:.4f}, {rmax:.4f}]")

# Mann-Whitney U tests between adjacent TI grades
print(f"\n🔬 Mann-Whitney U test p-values between adjacent TI grades:")
for g1, g2 in zip(ti_grades[:-1], ti_grades[1:]):
    u_stat, p_val = mannwhitneyu(scores_by_ti[int(g1)], scores_by_ti[int(g2)], alternative='two-sided')
    print(f"TI-{int(g1)} vs TI-{int(g2)}: p = {p_val:.4g}")

# Binary classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

binary_accuracy = accuracy_score(df['binary_true'], df['binary_pred'])
binary_precision = precision_score(df['binary_true'], df['binary_pred'])
binary_recall = recall_score(df['binary_true'], df['binary_pred'])
binary_f1 = f1_score(df['binary_true'], df['binary_pred'])
binary_auc = roc_auc_score(df['binary_true'], df['optimized_inflammation_score'])

print(f"\n🎯 Binary Classification Metrics (TI ≥ 2):")
print(f"Accuracy: {binary_accuracy:.3f}")
print(f"Precision: {binary_precision:.3f}")
print(f"Recall (Sensitivity): {binary_recall:.3f}")
print(f"F1-Score: {binary_f1:.3f}")
print(f"AUC: {binary_auc:.3f}")
print(f"Cohen's Kappa: {kappa_binary:.3f}")