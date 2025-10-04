import os
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

# Train Decision Tree with default parameters
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Decision Tree (Default): MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(dt, 'data/models/decision_tree_basic.pkl')
joblib.dump({'random_state': 42}, 'data/models/decision_tree_basic_params.pkl')
print("Saved Decision Tree basic model and parameters to data/models/")

# Plot actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Actual vs Predicted (Decision Tree Default)')
plt.tight_layout()
plt.show()
