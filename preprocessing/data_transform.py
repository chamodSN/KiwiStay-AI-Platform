# Fixed preprocessing.data_transform.py
# Changes:
# - Ensured numerical_cols_extended contains unique columns before scaling.
# - Sanitized polynomial feature names to avoid overlap with original features.
# - Moved duplicate check for numerical_cols_extended earlier to prevent duplicates in scaler.
# - Added explicit logging of numerical_cols_extended for debugging.
# - Kept existing duplicate checks and scaler saving intact.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
from common.config import REDUCTION_OUTPUT_CSV, TRANSFORM_OUTPUT_CSV
import joblib

np.random.seed(42)  # Seed for reproducibility

# Load data
df = pd.read_csv(REDUCTION_OUTPUT_CSV)

# Log transform price (skewed)
df['log_price'] = np.log1p(df['price'])

# Target encoding for 'neighbourhood'
te = TargetEncoder()
df['neighbourhood_encoded'] = te.fit_transform(
    df['neighbourhood'], df['log_price'])

# Define categorical columns
cat_cols = ['room_type_cleaned', 'neighbourhood_group_cleaned']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

# Define numerical columns
numerical_cols = [
    'latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm_cleaned',
    'is_inactive', 'recent_reviews_ratio', 'loc_pca1', 'loc_pca2'
]

# Add interaction features
inter_cols = ['availability_365', 'minimum_nights_cleaned',
              'calculated_host_listings_count', 'recent_reviews_ratio']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[inter_cols])
poly_cols = poly.get_feature_names_out(inter_cols)

# Sanitize polynomial feature names to avoid overlap with original features
# Prefix with 'poly_' and replace spaces
poly_cols = [f"poly_{col.replace(' ', '_')}" for col in poly_cols]
# Ensure unique column names
unique_poly_cols = []
seen = set()
for i, col in enumerate(poly_cols):
    if col in seen:
        unique_poly_cols.append(f"{col}_{i}")
    else:
        unique_poly_cols.append(col)
        seen.add(col)

poly_df = pd.DataFrame(poly_features, columns=unique_poly_cols, index=df.index)

# Check for duplicates in poly_df
if poly_df.columns.duplicated().any():
    duplicates = poly_df.columns[poly_df.columns.duplicated()].tolist()
    print(f"Warning: Duplicate polynomial columns detected: {duplicates}")
    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()].copy()

# Combine all
df_transformed = pd.concat([df.drop(
    ['neighbourhood_group_cleaned', 'neighbourhood'], axis=1), encoded_df, poly_df], axis=1)

# Check for duplicates in df_transformed
if df_transformed.columns.duplicated().any():
    duplicates = df_transformed.columns[df_transformed.columns.duplicated(
    )].tolist()
    print(f"Warning: Duplicate columns in df_transformed: {duplicates}")
    df_transformed = df_transformed.loc[:, ~
                                        df_transformed.columns.duplicated()].copy()

# Define numerical columns for scaling (ensure no duplicates)
numerical_cols_extended = numerical_cols + \
    poly_df.columns.tolist() + ['neighbourhood_encoded']
numerical_cols_extended = pd.Index(
    numerical_cols_extended).unique().tolist()  # Ensure unique columns
print(f"Columns to scale (numerical_cols_extended): {numerical_cols_extended}")

# Scale numerical features
scaler = MinMaxScaler()
df_transformed[numerical_cols_extended] = scaler.fit_transform(
    df_transformed[numerical_cols_extended])

# Save scaler for frontend use
os.makedirs('data/processed', exist_ok=True)
joblib.dump(scaler, 'data/processed/scaler.pkl')
print("Saved scaler to data/processed/scaler.pkl")

# Verify scaler feature names
if hasattr(scaler, 'feature_names_in_'):
    if pd.Index(scaler.feature_names_in_).duplicated().any():
        duplicates = scaler.feature_names_in_[
            pd.Index(scaler.feature_names_in_).duplicated()].tolist()
        print(
            f"Warning: Duplicate columns in scaler.feature_names_in_: {duplicates}")
        scaler.feature_names_in_ = pd.Index(
            scaler.feature_names_in_).unique().tolist()
    print(f"Scaler feature names: {scaler.feature_names_in_}")

print("Shape after transformation:", df_transformed.shape)

# Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['price'], kde=True, ax=ax[0]).set_title('Original Price')
sns.histplot(df['log_price'], kde=True, ax=ax[1]).set_title('Log Price')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['neighbourhood_encoded'], kde=True, ax=ax[0]
             ).set_title('Neighbourhood Target Encoding')
sns.histplot(df['recent_reviews_ratio'], kde=True, ax=ax[1]).set_title(
    'Recent Reviews Ratio (New Interaction Base)')
plt.show()

# Save transformed data
df_transformed.to_csv(TRANSFORM_OUTPUT_CSV, index=False)
