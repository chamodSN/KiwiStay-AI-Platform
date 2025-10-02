# Code for Data Reduction
# Explanation: Reduce dimensionality: Drop irrelevant columns (e.g., name, host_name, last_review if not useful for prediction).
# Use PCA for numerical features if high dims, but here features are few.
# For other ML: Keep lat/long for clustering (location-based).
# Output: Show reduced shape, variance explained if PCA.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from common.config import REDUCTION_OUTPUT_CSV, CLEANING_OUTPUT_CSV

df = pd.read_csv(CLEANING_OUTPUT_CSV)

# Drop irrelevant columns for prediction (name, host_name, last_review, license if still there)
# Assuming not useful for price prediction
drop_cols = ['id', 'host_id', 'name', 'host_name', 'last_review']

df_reduced = df.drop(drop_cols, axis=1)

# Numerical features for PCA Principal Component Analysis
numerical_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                  'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm']
X_num = df_reduced[numerical_cols]

# Standardize and apply PCA (reduce to 3 components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Variance explained by PCA:", pca.explained_variance_ratio_)

# Add PCA components back
df_reduced[['pca1', 'pca2', 'pca3']] = X_pca

print("Shape after reduction:", df_reduced.shape)

# Plot PCA variance
plt.figure(figsize=(6, 4))
plt.bar(range(1, 4), pca.explained_variance_ratio_)
plt.title('PCA Variance Explained')
plt.show()

df_reduced.to_csv(REDUCTION_OUTPUT_CSV, index=False)
