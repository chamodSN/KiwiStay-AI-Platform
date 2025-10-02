from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import os

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized_enhanced.csv')
X = df.select_dtypes(include=['float64', 'int64', 'uint8']).drop(
    ['price', 'log_price', 'popularity_bin', 'availability_bin_high_low'], axis=1, errors='ignore')
y = df['popularity_bin']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Basic Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print(
    f"Random Forest (Basic): Accuracy={acc:.2f}, F1={f1:.2f}, ROC-AUC={roc_auc:.2f}")

# Save model
os.makedirs('data/models/classification', exist_ok=True)
joblib.dump(model, 'data/models/classification/rf_basic.pkl')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
