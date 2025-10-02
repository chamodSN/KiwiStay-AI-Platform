
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
from common.config import REDUCTION_OUTPUT_CSV, TRANSFORM_OUTPUT_CSV

df = pd.read_csv(REDUCTION_OUTPUT_CSV)

# Normalize/scale numerical features, encode categoricals with target encoding for high-cardinality 'neighbourhood',
# add interaction features for top important columns, and extract text features from 'name'.
# Log transform skewed price.
# For other ML: One-hot for association rules, normalization for clustering.
# Scale after engineering to include new features.
# Output: Show transformed data, plots before/after for price and new features.

# Log transform price (skewed)
# Use log_price as target for regression
df['log_price'] = np.log1p(df['price'])

# Target encoding for high-cardinality 'neighbourhood' (instead of one-hot)
te = TargetEncoder()
df['neighbourhood_encoded'] = te.fit_transform(
    df['neighbourhood'], df['log_price'])

# Define categorical columns (exclude 'neighbourhood' due to target encoding)
cat_cols = ['room_type_cleaned', 'neighbourhood_group_cleaned']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

# Define numerical columns (include new features from cleaning: days_since_last_review, is_inactive)
numerical_cols = ['latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm',
                  'is_inactive']

# Add interaction features for top important columns
# Interactions capture relationships between features that a linear model might miss.
inter_cols = ['availability_365', 'minimum_nights_cleaned',
              'calculated_host_listings_count']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[inter_cols])
poly_df = pd.DataFrame(
    poly_features, columns=poly.get_feature_names_out(inter_cols), index=df.index)

# Combine all features: drop original categoricals, keep numerical, add encoded and interaction features
df_transformed = pd.concat([df.drop(['neighbourhood_group_cleaned', 'neighbourhood'], axis=1), encoded_df, poly_df], axis=1)  # Keep 'room_type_cleaned'
# Scale numerical columns (including new features) after engineering
scaler = MinMaxScaler()
numerical_cols_extended = numerical_cols + \
    poly.get_feature_names_out(inter_cols).tolist() + ['neighbourhood_encoded']
df_transformed[numerical_cols_extended] = scaler.fit_transform(
    df_transformed[numerical_cols_extended])

print("Shape after transformation:", df_transformed.shape)

# Plot before/after log price
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['price'], kde=True, ax=ax[0]).set_title('Original Price')
sns.histplot(df['log_price'], kde=True, ax=ax[1]).set_title('Log Price')
plt.show()

# Plot distribution of new features
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['neighbourhood_encoded'], kde=True, ax=ax[0]
             ).set_title('Neighbourhood Target Encoding')
plt.show()

df_transformed.to_csv(TRANSFORM_OUTPUT_CSV, index=False)
