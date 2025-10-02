# Code for Data Reduction
# Explanation: Reduce dimensionality: Drop irrelevant columns (e.g., name, host_name, last_review if not useful for prediction).
# Use PCA for numerical features if high dims, but here features are few.
# For other ML: Keep lat/long for clustering (location-based).
# Output: Show reduced shape, variance explained if PCA.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common.config import REDUCTION_OUTPUT_CSV, CLEANING_OUTPUT_CSV

np.random.seed(42)  # Seed for reproducibility

df = pd.read_csv(CLEANING_OUTPUT_CSV)

# Drop irrelevant columns for prediction (name, host_name, last_review, license if still there)
drop_cols = ['id', 'host_id', 'name', 'host_name', 'last_review']
df_reduced = df.drop(drop_cols, axis=1)

# Numerical features for general PCA
numerical_cols = ['minimum_nights_cleaned', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm_cleaned',
                  'recent_reviews_ratio']  # UPDATE: Added new features

X_num = df_reduced[numerical_cols]

# Standardize and apply PCA (reduce to 3 components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Variance explained by general PCA:", pca.explained_variance_ratio_)

# Add PCA components back
df_reduced[['pca1', 'pca2', 'pca3']] = X_pca

# UPDATE: Separate PCA for location (lat/long) for clustering (reduces to 2D for spatial segments)
location_cols = ['latitude', 'longitude']
X_loc = df_reduced[location_cols]
X_loc_scaled = scaler.fit_transform(X_loc)
loc_pca = PCA(n_components=2, random_state=42)
X_loc_pca = loc_pca.fit_transform(X_loc_scaled)
print("Variance explained by location PCA:", loc_pca.explained_variance_ratio_)

# Add location PCA
df_reduced[['loc_pca1', 'loc_pca2']] = X_loc_pca
# WHY: Aids clustering for location-based segments ("urban" vs "rural"); preserves for Folium maps without high dims. Doesn't affect regression.

print("Shape after reduction:", df_reduced.shape)

# Plot general PCA variance
plt.figure(figsize=(6, 4))
plt.bar(range(1, 4), pca.explained_variance_ratio_)
plt.title('General PCA Variance Explained')
plt.show()

# Plot location PCA variance
plt.figure(figsize=(6, 4))
plt.bar(range(1, 3), loc_pca.explained_variance_ratio_)
plt.title('Location PCA Variance Explained')
plt.show()

df_reduced.to_csv(REDUCTION_OUTPUT_CSV, index=False)
