# Code for Random Forest Regressor (Basic Parameters)
# Explanation: Default parameters, save model and parameters, evaluate, and plot.

import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

# Train model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest (Default): MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(rf, 'data/models/random_forest_basic.pkl')
joblib.dump({'random_state': 42}, 'data/models/random_forest_basic_params.pkl')
print("Saved Random Forest basic model and parameters to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Random Forest Default)')
plt.tight_layout()
plt.show()