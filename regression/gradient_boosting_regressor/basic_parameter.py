import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import os

# Explanation: Train Gradient Boosting with default parameters as a baseline.
# Save model for access by other scripts (e.g., clustering, Streamlit).
# Evaluate and plot performance.

# Load data
X_train = joblib.load('data/processed/X_train.pkl')
y_train = joblib.load('data/processed/y_train.pkl')
X_test = joblib.load('data/processed/X_test.pkl')
y_test = joblib.load('data/processed/y_test.pkl')

# Train Gradient Boosting with default parameters
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Gradient Boosting (Default): MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(gb, 'data/models/gradient_boosting_basic.pkl')
joblib.dump({'random_state': 42}, 'data/models/gradient_boosting_basic_params.pkl')
print("Saved Gradient Boosting basic model and parameters to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Gradient Boosting Default)')
plt.show()