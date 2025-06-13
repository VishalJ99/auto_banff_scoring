import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, cohen_kappa_score, roc_auc_score,
                           precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
csv_path = "data_handling/cluster_metrics_eps37.csv"
df = pd.read_csv(csv_path)

# Drop missing values
df = df.dropna(subset=["TI", "inflammatory_per_cortex_patch", "wbc_clustering_index"])

# Prepare features
X = df[['inflammatory_per_cortex_patch', 'wbc_clustering_index']].values
y_multiclass = df['TI'].astype(int).values
y_binary = (df['TI'] >= 2).astype(int).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dataset Summary:")
print(f"Total samples: {len(df)}")
print(f"TI distribution: {np.bincount(y_multiclass)}")
print(f"Binary distribution (TI≥2): {np.bincount(y_binary)}")

def evaluate_multiclass_classification(X, y):
    """Evaluate multi-class classification (TI 0,1,2,3)"""
    print("\n" + "="*50)
    print("MULTI-CLASS CLASSIFICATION (TI 0,1,2,3)")
    print("="*50)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_kappa = cross_val_score(model, X, y, cv=cv, 
                              scoring='cohen_kappa' if hasattr(cv, 'cohen_kappa') else None)
    
    print(f"Cross-validation Accuracy: {cv_accuracy.mean():.3f} (±{cv_accuracy.std():.3f})")
    
    # Train on full dataset for detailed analysis
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    
    print(f"\nFull Dataset Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['TI-0', 'TI-1', 'TI-2', 'TI-3']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['TI-0', 'TI-1', 'TI-2', 'TI-3'],
                yticklabels=['TI-0', 'TI-1', 'TI-2', 'TI-3'])
    plt.title(f'Multi-class Confusion Matrix\n(Accuracy={accuracy:.3f}, κ={kappa:.3f})')
    plt.xlabel('Predicted TI Grade')
    plt.ylabel('True TI Grade')
    plt.tight_layout()
    plt.savefig('multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature coefficients
    feature_names = ['Normalised Cell Count', 'WBC Clustering Index']
    print(f"\nLogistic Regression Coefficients:")
    for i, class_name in enumerate(['TI-0', 'TI-1', 'TI-2', 'TI-3']):
        print(f"{class_name}: {dict(zip(feature_names, model.coef_[i]))}")
    
    return accuracy, kappa, model

def evaluate_binary_classification(X, y):
    """Evaluate binary classification (TI < 2 vs TI ≥ 2)"""
    print("\n" + "="*50)
    print("BINARY CLASSIFICATION (TI < 2 vs TI ≥ 2)")
    print("="*50)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"Cross-validation Accuracy: {cv_accuracy.mean():.3f} (±{cv_accuracy.std():.3f})")
    print(f"Cross-validation AUC: {cv_auc.mean():.3f} (±{cv_auc.std():.3f})")
    
    # Train on full dataset for detailed analysis
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary')
    
    print(f"\nFull Dataset Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.3f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['TI < 2', 'TI ≥ 2']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['TI < 2', 'TI ≥ 2'],
                yticklabels=['TI < 2', 'TI ≥ 2'])
    plt.title(f'Binary Classification Confusion Matrix\n(Accuracy={accuracy:.3f}, κ={kappa:.3f}, AUC={auc:.3f})')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature coefficients
    feature_names = ['Normalised Cell Count', 'WBC Clustering Index']
    print(f"\nLogistic Regression Coefficients:")
    coefficients = dict(zip(feature_names, model.coef_[0]))
    for feature, coef in coefficients.items():
        print(f"{feature}: {coef:.3f}")
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Binary Classification (TI ≥ 2)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('binary_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, kappa, auc, model

def compare_with_threshold_method(X, y_binary):
    """Compare logistic regression with simple threshold-based classification"""
    print("\n" + "="*30)
    print("COMPARISON WITH THRESHOLD METHOD")
    print("="*30)
    
    # Create combined score (same as your previous analysis)
    alpha, beta = 0.56, 0.44
    z_norm = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    z_wci = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    combined_score = alpha * z_norm + beta * z_wci
    
    # Threshold-based classification (score >= 0)
    y_pred_threshold = (combined_score >= 0).astype(int)
    
    # Calculate threshold method metrics
    acc_threshold = accuracy_score(y_binary, y_pred_threshold)
    kappa_threshold = cohen_kappa_score(y_binary, y_pred_threshold)
    auc_threshold = roc_auc_score(y_binary, combined_score)
    
    print(f"Threshold Method (score ≥ 0):")
    print(f"  Accuracy: {acc_threshold:.3f}")
    print(f"  Cohen's Kappa: {kappa_threshold:.3f}")
    print(f"  AUC: {auc_threshold:.3f}")
    
    # Train logistic regression for comparison
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_scaled, y_binary)
    y_pred_lr = lr.predict(X_scaled)
    y_proba_lr = lr.predict_proba(X_scaled)[:, 1]
    
    acc_lr = accuracy_score(y_binary, y_pred_lr)
    kappa_lr = cohen_kappa_score(y_binary, y_pred_lr)
    auc_lr = roc_auc_score(y_binary, y_proba_lr)
    
    print(f"\nLogistic Regression:")
    print(f"  Accuracy: {acc_lr:.3f}")
    print(f"  Cohen's Kappa: {kappa_lr:.3f}")
    print(f"  AUC: {auc_lr:.3f}")
    
    return acc_threshold, kappa_threshold, auc_threshold, acc_lr, kappa_lr, auc_lr

# Run all analyses
print("Running Classification Analysis...")
print("Features: Normalised Cell Count + WBC Clustering Index")

# Multi-class classification
mc_accuracy, mc_kappa, mc_model = evaluate_multiclass_classification(X_scaled, y_multiclass)

# Binary classification  
bin_accuracy, bin_kappa, bin_auc, bin_model = evaluate_binary_classification(X_scaled, y_binary)

# Comparison with threshold method
thresh_acc, thresh_kappa, thresh_auc, lr_acc, lr_kappa, lr_auc = compare_with_threshold_method(X, y_binary)

# Summary
print("\n" + "="*50)
print("SUMMARY OF RESULTS")
print("="*50)
print(f"Multi-class Classification:")
print(f"  Accuracy: {mc_accuracy:.3f}")
print(f"  Cohen's Kappa: {mc_kappa:.3f}")
print(f"\nBinary Classification (Logistic Regression):")
print(f"  Accuracy: {bin_accuracy:.3f}")
print(f"  Cohen's Kappa: {bin_kappa:.3f}")
print(f"  AUC: {bin_auc:.3f}")
print(f"\nBinary Classification (Threshold Method):")
print(f"  Accuracy: {thresh_acc:.3f}")
print(f"  Cohen's Kappa: {thresh_kappa:.3f}")
print(f"  AUC: {thresh_auc:.3f}")