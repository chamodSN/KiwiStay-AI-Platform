import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# UPDATE: Added Classifier for binary targets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# UPDATE: Added classif
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib
import os
from common.config import DISCRETIZATION_OUTPUT_CSV

np.random.seed(42)  # Seed for reproducibility

# Explanation: Use correlation, mutual info, and RF feature importance to check relation to price.
# Create and save X_train, y_train for other models to access.
# Plots: Correlation heatmap, scatter for relationships.
# UPDATE: Added feature engineering for classification target 'popularity_bin' (high/low based on reviews_per_month > median).
# UPDATE: Added 'availability_bin_high_low' (high if availability_365 > median).
# UPDATE: Compute importance for multiple targets (log_price, popularity_bin, availability_bin_high_low).
# WHY: Supports demand prediction (classification) and availability forecasting; RFClassifier for binary targets.

df_transformed = pd.read_csv(DISCRETIZATION_OUTPUT_CSV)

# New targets
median_reviews = df_transformed['reviews_per_month'].median()
df_transformed['popularity_bin'] = np.where(
    # Binary for classification
    df_transformed['reviews_per_month'] > median_reviews, 1, 0)

median_avail = df_transformed['availability_365'].median()
df_transformed['availability_bin_high_low'] = np.where(
    df_transformed['availability_365'] > median_avail, 1, 0)

# Prepare data
df_ml = df_transformed.select_dtypes(
    include=['float64', 'int64', 'uint8']).dropna().reset_index(drop=True)
df_ml = df_ml.drop(['id', 'host_id', 'days_since_last_review'],
                   axis=1, errors='ignore')  # Explicit drop

# For regression (log_price)
X_reg = df_ml.drop(['price', 'log_price', 'popularity_bin',
                   'availability_bin_high_low'], axis=1, errors='ignore')
y_reg = df_ml['log_price']

# For classification (popularity_bin)
X_clf = X_reg.copy()  # Same features
y_clf = df_ml['popularity_bin']

# For availability forecasting (availability_bin_high_low)
y_avail = df_ml['availability_bin_high_low']

# Split (shared X for all)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42)
_, _, y_avail_train, y_avail_test = train_test_split(
    X_clf, y_avail, test_size=0.2, random_state=42)

# Save
os.makedirs('data/processed', exist_ok=True)
joblib.dump(X_train, 'data/processed/X_train.pkl')
joblib.dump(y_reg_train, 'data/processed/y_reg_train.pkl')  # Regression
joblib.dump(y_clf_train, 'data/processed/y_clf_train.pkl')  # Classification
joblib.dump(y_avail_train, 'data/processed/y_avail_train.pkl')  # Availability
joblib.dump(X_test, 'data/processed/X_test.pkl')
joblib.dump(y_reg_test, 'data/processed/y_reg_test.pkl')
joblib.dump(y_clf_test, 'data/processed/y_clf_test.pkl')
joblib.dump(y_avail_test, 'data/processed/y_avail_test.pkl')
print("Saved train/test sets for regression, classification, and availability to data/processed/")

# Correlation for regression
corr = df_ml.corr()['log_price'].sort_values(ascending=False)
print("Correlations with log_price:\n", corr)

# Mutual info for regression
mi_reg = mutual_info_regression(X_reg, y_reg, random_state=42)
mi_df_reg = pd.Series(mi_reg, index=X_reg.columns).sort_values(ascending=False)
print("\nMutual Info (Regression):\n", mi_df_reg)

# RF importance for regression
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg, y_reg)
imp_reg = pd.Series(rf_reg.feature_importances_,
                    index=X_reg.columns).sort_values(ascending=False)
print("\nRF Importance (Regression):\n", imp_reg)

# Mutual info for classification
mi_clf = mutual_info_classif(X_clf, y_clf, random_state=42)
mi_df_clf = pd.Series(mi_clf, index=X_clf.columns).sort_values(ascending=False)
print("\nMutual Info (Classification - Popularity):\n", mi_df_clf)

# RF importance for classification
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_clf, y_clf)
imp_clf = pd.Series(rf_clf.feature_importances_,
                    index=X_clf.columns).sort_values(ascending=False)
print("\nRF Importance (Classification - Popularity):\n", imp_clf)

# Mutual info for availability
mi_avail = mutual_info_classif(X_clf, y_avail, random_state=42)
mi_df_avail = pd.Series(
    mi_avail, index=X_clf.columns).sort_values(ascending=False)
print("\nMutual Info (Availability Forecasting):\n", mi_df_avail)

# RF importance for availability
rf_avail = RandomForestClassifier(random_state=42)
rf_avail.fit(X_clf, y_avail)
imp_avail = pd.Series(rf_avail.feature_importances_,
                      index=X_clf.columns).sort_values(ascending=False)
print("\nRF Importance (Availability Forecasting):\n", imp_avail)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_ml.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter example: log_price vs availability
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_ml['availability_365'], y=df_ml['log_price'])
plt.title('Availability vs Log Price')
plt.show()

# Additional scatter for new target: popularity_bin vs recent_reviews_ratio
plt.figure(figsize=(6, 4))
sns.boxplot(x=df_ml['popularity_bin'], y=df_ml['recent_reviews_ratio'])
plt.title('Popularity Bin vs Recent Reviews Ratio')
plt.show()

# Decision: Features have non-linear relationships; ML is worth it for all tasks.
print("\nML Worth It: Yes, due to multiple influencing features with non-linear relationships.")
