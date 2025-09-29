import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import joblib
import os
from common.config import INPUT_CSV

# Explanation: Use correlation, mutual info, and RF feature importance to check relation to price.
# Create and save X_train, y_train for other models to access.
# Plots: Correlation heatmap, scatter for relationships.
# Fix ValueError by ensuring df_ml has 1D columns, checking for duplicates, and aligning indices.

# Load transformed data
df_transformed = pd.read_csv(INPUT_CSV)

# Prepare data (use transformed df, drop non-numeric)
# Ensure indices are aligned and duplicates are removed
df_ml = df_transformed.select_dtypes(include=['float64', 'int64', 'uint8']).dropna().reset_index(drop=True)

# Check for duplicate columns
duplicate_cols = df_ml.columns[df_ml.columns.duplicated()]
if len(duplicate_cols) > 0:
    print("Duplicate columns found:", duplicate_cols)
    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()]  # Keep first occurrence

# Verify column shapes for plotting
print("df_ml shape:", df_ml.shape)
print("df_ml columns:", df_ml.columns.tolist())
print("NaNs in df_ml:\n", df_ml.isnull().sum())
print("availability_365 shape:", df_ml['availability_365'].shape)
print("log_price shape:", df_ml['log_price'].shape)

# Define features and target
X = df_ml.drop(['price', 'log_price'], axis=1, errors='ignore')  # Features
y = df_ml['log_price'] if 'log_price' in df_ml else df_ml['price']

# Split into train/test (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Save X_train, y_train, X_test, y_test to disk for other models
os.makedirs('data/processed', exist_ok=True)
joblib.dump(X_train, 'data/processed/X_train.pkl')
joblib.dump(y_train, 'data/processed/y_train.pkl')
joblib.dump(X_test, 'data/processed/X_test.pkl')
joblib.dump(y_test, 'data/processed/y_test.pkl')
print("Saved X_train, y_train, X_test, y_test to data/processed/")

# Correlation
corr = df_ml.corr()['log_price'].sort_values(ascending=False)
print("Correlations with log_price:\n", corr)

# Mutual info
mi = mutual_info_regression(X, y, random_state=42)
mi_df = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("\nMutual Info:\n", mi_df)

# RF importance
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRF Importance:\n", imp)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_ml.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter example: price vs availability (validate columns)
if 'availability_365' in df_ml.columns and 'log_price' in df_ml.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df_ml['availability_365'], y=df_ml['log_price'])
    plt.title('Availability vs Log Price')
    plt.show()
else:
    print("Error: 'availability_365' or 'log_price' missing in df_ml")

# Decision: Features like room_type, location have varying importance – ML worth it
print("\nML Worth It: Yes, due to multiple influencing features with non-linear relationships.")