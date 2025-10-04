#stacking_ensemble.py

import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Explanation: Stack optimized models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
# Load models from data/models/ and data from data/processed/.
# Save stacking model for reuse.

# Load data
X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

# Load optimized models
best_lr = joblib.load('data/models/linear_regression_optimized.pkl')
best_dt = joblib.load('data/models/decision_tree_optimized.pkl')
best_rf = joblib.load('data/models/random_forest_optimized.pkl')
best_gb = joblib.load('data/models/gradient_boosting_optimized.pkl')
best_xg = joblib.load('data/models/xgboost_optimized.pkl')

# Define stacking ensemble
estimators = [
    ('lr', best_lr),
    ('dt', best_dt),
    ('rf', best_rf),
    ('gb', best_gb),
    ('xg', best_xg)
]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Stacking Ensemble: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model
os.makedirs('data/models', exist_ok=True)
joblib.dump(stack, 'data/models/stacking_ensemble.pkl')
print("Saved Stacking Ensemble model to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Stacking Ensemble)')
plt.show()