import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ------------------- Load Data -------------------
df = pd.read_csv("/data2/ac2220/auto_banff_scoring/data_processing/combined_metrics_output_demo.csv")

# Drop rows with missing TI
df = df[df["TI"].notna()].copy()
df["TI"] = df["TI"].astype(int)

# Features to use
features = ["inflammatory_per_cortex_patch", "wbc_clustering_index"]
X = df[features]

# ------------------- Multiclass Model (TI) -------------------
print("\n🔢 Training Multiclass Logistic Regression (TI prediction)...")

y_multi = df["TI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, stratify=y_multi, test_size=0.2, random_state=42
)

scaler_multi = StandardScaler()
X_train_scaled = scaler_multi.fit_transform(X_train)
X_test_scaled = scaler_multi.transform(X_test)

clf_multi = LogisticRegression(max_iter=1000, multi_class="multinomial")
clf_multi.fit(X_train_scaled, y_train)

y_pred_multi = clf_multi.predict(X_test_scaled)

print("\n📊 Multiclass Classification Report:")
print(classification_report(y_test, y_pred_multi, digits=3))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_multi))

# Save model and scaler
joblib.dump(clf_multi, "logreg_multiclass.pkl")
joblib.dump(scaler_multi, "scaler_multiclass.pkl")

# ------------------- Binary Model (TI ≥ 2) -------------------
print("\n🟩 Training Binary Logistic Regression (TI ≥ 2 classification)...")

y_binary = (df["TI"] >= 2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, stratify=y_binary, test_size=0.2, random_state=42
)

scaler_bin = StandardScaler()
X_train_scaled = scaler_bin.fit_transform(X_train)
X_test_scaled = scaler_bin.transform(X_test)

clf_bin = LogisticRegression(max_iter=1000)
clf_bin.fit(X_train_scaled, y_train)

y_pred_bin = clf_bin.predict(X_test_scaled)

print("\n📊 Binary Classification Report:")
print(classification_report(y_test, y_pred_bin, digits=3))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bin))

# Save model and scaler
joblib.dump(clf_bin, "logreg_binary.pkl")
joblib.dump(scaler_bin, "scaler_binary.pkl")

print("\n✅ Models saved: logreg_multiclass.pkl, logreg_binary.pkl")
