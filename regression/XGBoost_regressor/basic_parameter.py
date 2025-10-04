# Code for XGBoost Regressor (Basic Parameters)
# Explanation: Default.

import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train = joblib.load('data/processed/X_reg_train.pkl')
y_train = joblib.load('data/processed/y_reg_train.pkl')
X_test = joblib.load('data/processed/X_reg_test.pkl')
y_test = joblib.load('data/processed/y_reg_test.pkl')

xg = xgb.XGBRegressor(random_state=42)
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost (Default): MAE={mae:.2f}, MSE={mse:.2f}, R2={r2:.2f}")

# Save model and parameters
os.makedirs('data/models', exist_ok=True)
joblib.dump(xg, 'data/models/xgboost_basic.pkl')
joblib.dump({'random_state': 42}, 'data/models/xgboost_basic_params.pkl')
print("Saved XGBoost basic model and parameters to data/models/")

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted (XGBoost)')
plt.show()