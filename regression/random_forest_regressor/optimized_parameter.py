# Code for Random Forest Regressor (Optimized Parameters)
# Explanation: Optimize parameters using GridSearchCV, save model and best parameters,
# evaluate, and compare with default in side-by-side plots.

import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# Train with GridSearchCV
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_opt = best_rf.predict(X_test)

# Evaluate
mae_opt = mean_absolute_error(y_test, y_pred_opt)
mse_opt = mean_squared_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)
print(f"Random Forest (Optimized): Params={grid.best_params_}, MAE={mae_opt:.2f}, MSE={mse_opt:.2f}, R2={r2_opt:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(best_rf, 'data/models/random_forest_optimized.pkl')
joblib.dump(grid.best_params_, 'data/models/random_forest_optimized_params.pkl')
print("Saved Random Forest optimized model and parameters to data/models/")

# Plot actual vs predicted (default vs optimized)
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(y_test, y_pred_default)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_xlabel('Actual Log Price')
ax[0].set_ylabel('Predicted Log Price')
ax[0].set_title('Random Forest Default')
ax[1].scatter(y_test, y_pred_opt)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[1].set_xlabel('Actual Log Price')
ax[1].set_ylabel('Predicted Log Price')
ax[1].set_title('Random Forest Optimized')
plt.tight_layout()
plt.show()