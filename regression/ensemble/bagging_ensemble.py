
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Explanation: Bagging with DecisionTree as base estimator.
# Load data from data/processed/ and save model for reuse.

# Load data
X_train = joblib.load('data/processed/X_train.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
X_test = joblib.load('data/processed/X_test.pkl')
y_test = joblib.load('data/processed/y_test.pkl')

# Train Bagging with DecisionTree
bag = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), n_estimators=100, random_state=42)
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Bagging Ensemble: MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(bag, 'data/models/bagging_ensemble.pkl')
joblib.dump({'estimator': 'DecisionTreeRegressor', 'n_estimators': 100, 'random_state': 42}, 'data/models/bagging_ensemble_params.pkl')
print("Saved Bagging Ensemble model and parameters to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Bagging Ensemble)')
plt.show()