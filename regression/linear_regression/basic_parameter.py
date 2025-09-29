# Code for Linear Regression (Basic Parameters)
# Explanation: Simple Linear Regression with default parameters.
# Metrics: MAE, MSE, R2.
# Save model and parameters using joblib.
# Plot: Actual vs Predicted.

import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = joblib.load('data/processed/X_train.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
X_test = joblib.load('data/processed/X_test.pkl')
y_test = joblib.load('data/processed/y_test.pkl')

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression (Default): MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(lr, 'data/models/linear_regression_basic.pkl')
joblib.dump({'fit_intercept': True}, 'data/models/linear_regression_basic_params.pkl')
print("Saved Linear Regression basic model and parameters to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Linear Regression)')
plt.tight_layout()
plt.show()