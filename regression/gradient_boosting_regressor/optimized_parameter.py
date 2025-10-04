import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import os

# Explanation: Optimize Gradient Boosting parameters using GridSearchCV.
# Save model and best parameters for access by other scripts.
# Evaluate and compare with default.

# Load data
X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

# Define parameter grid (expanded for better optimization)
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

# Train with GridSearchCV
grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)
best_gb = grid.best_estimator_
y_pred_opt = best_gb.predict(X_test)

# Evaluate
mae_opt = mean_absolute_error(y_test, y_pred_opt)
mse_opt = mean_squared_error(y_test, y_pred_opt)
r2_opt = r2_score(y_test, y_pred_opt)
print(f"Gradient Boosting (Optimized): Params={grid.best_params_}, MAE={mae_opt:.2f}, MSE={mse_opt:.2f}, R2={r2_opt:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(best_gb, 'data/models/gradient_boosting_optimized.pkl')
joblib.dump(grid.best_params_, 'data/models/gradient_boosting_optimized_params.pkl')
print("Saved Gradient Boosting optimized model and parameters to data/models/")

# Plot actual vs predicted (default vs optimized)
gb_default = GradientBoostingRegressor(random_state=42)
gb_default.fit(X_train, y_train)
y_pred_default = gb_default.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(y_test, y_pred_default)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_xlabel('Actual Log Price')
ax[0].set_ylabel('Predicted Log Price')
ax[0].set_title('Gradient Boosting Default')
ax[1].scatter(y_test, y_pred_opt)
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[1].set_xlabel('Actual Log Price')
ax[1].set_ylabel('Predicted Log Price')
ax[1].set_title('Gradient Boosting Optimized')
plt.tight_layout()
plt.show()