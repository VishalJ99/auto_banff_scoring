import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("/data2/ac2220/data_handling/combined_metrics_output.csv")

# Select features and target
features = ["inflammatory_per_cortex_patch", "wbc_clustering_index"]
X = df[features]
y = df["TI"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
clf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)

print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

print("📉 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))