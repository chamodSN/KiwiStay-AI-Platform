# Fixed classification.XGBoost.xgb_optimized.py
# Changes: Load task-specific processed data from PKL files instead of recreating X from CSV to ensure consistency with reduced features (no neighbourhood_group one-hot columns).

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import os

np.random.seed(42)

# Load pre-saved task-specific data (from feature_importance.py)
try:
    X_train = joblib.load('data/processed/X_clf_train.pkl')
    y_train = joblib.load('data/processed/y_clf_train.pkl')
    X_test = joblib.load('data/processed/X_clf_test.pkl')
    y_test = joblib.load('data/processed/y_clf_test.pkl')
    print(
        f"Loaded classification data: Train shape {X_train.shape}, Test shape {X_test.shape}")
except FileNotFoundError:
    print("Task-specific data not found. Run feature_importance.py first.")
    exit(1)

# Fix duplicate columns (common issue from polynomial features)
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

print(f"After duplicate removal - Train columns: {X_train.columns.tolist()}")

# Ensure all features are numeric and fill NaNs
X_train = X_train.select_dtypes(include=[np.number]).fillna(0).astype(float)
X_test = X_test.select_dtypes(include=[np.number]).fillna(0).astype(float)

# Ensure y is integer binary
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f"Final shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

# GridSearch for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
grid = GridSearchCV(XGBClassifier(
    random_state=42, eval_metric='logloss'), param_grid, scoring='roc_auc', cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print(
    f"XGBoost Popularity (Optimized): Best Params={grid.best_params_}, Accuracy={acc:.2f}, F1={f1:.2f}, ROC-AUC={roc_auc:.2f}")

# Save model
os.makedirs('data/models/classification', exist_ok=True)
joblib.dump(best_model, 'data/models/classification/xgb_optimized.pkl')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('XGBoost Optimized ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Optimized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
