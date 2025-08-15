# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_curve
)

# 1. Load dataset
data = pd.read_csv('creditcard.csv')

# 2. Feature & target separation
X = data.drop(['Class'], axis=1)
y = data['Class']

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict_proba(X_test)[:, 1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print(f"✅ Decision Tree ROC-AUC Score: {roc_auc_dt:.3f}")

# 6. Train SVM
svm = SVC(kernel='rbf', C=1.0, probability=False)
svm.fit(X_train, y_train)
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print(f"✅ SVM ROC-AUC Score: {roc_auc_svm:.3f}")

# 7. ROC Curve Plot
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', color='blue')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.3f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title('📈 ROC Curve: Decision Tree vs SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Binary class predictions for metrics
y_pred_dt_bin = (y_pred_dt >= 0.5).astype(int)
y_pred_svm_bin = (y_pred_svm >= 0.0).astype(int)

# 9. Classification Reports
print("\n📊 Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt_bin))

print("\n📊 SVM Classification Report:")
print(classification_report(y_test, y_pred_svm_bin))

# 10. Confusion Matrices
cm_dt = confusion_matrix(y_test, y_pred_dt_bin)
cm_svm = confusion_matrix(y_test, y_pred_svm_bin)

# Decision Tree Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("🧾 Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# SVM Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("🧾 SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 11. Optimal Threshold for DT (Bonus)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_dt)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\n🎯 Best Threshold for DT (F1-Score Maximized): {best_threshold:.2f}")
