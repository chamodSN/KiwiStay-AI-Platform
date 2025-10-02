import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized.csv')  # Use enhanced for consistency
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# FIX: Downsample to avoid memory error
n_sample = 2000  # Adjust as needed
if len(X_cluster) > n_sample:
    sample_idx = np.random.choice(len(X_cluster), n_sample, replace=False)
    X_cluster_sample = X_cluster.iloc[sample_idx]
    df_sample = df.iloc[sample_idx]
    print(f"Downsampled to {n_sample} samples for memory efficiency.")
else:
    X_cluster_sample = X_cluster
    df_sample = df

# Custom scorer for GridSearch (max silhouette)
def sil_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return -1

# GridSearch (limited params for speed/memory; affinity='nearest_neighbors' to sparsify)
param_grid = {
    'n_clusters': range(2, 6),  # FIX: Limit to 2-5 to reduce computation (full 2-11 too slow)
    'affinity': ['nearest_neighbors']  # FIX: Force sparse affinity
}
grid = GridSearchCV(SpectralClustering(random_state=42, n_neighbors=10), param_grid, scoring=sil_scorer, cv=3)
grid.fit(X_cluster_sample)
best_spectral = grid.best_estimator_
labels_opt = best_spectral.fit_predict(X_cluster_sample)

# Evaluate
unique_opt = len(set(labels_opt))
if unique_opt > 1:
    sil_opt = silhouette_score(X_cluster_sample, labels_opt)
    db_opt = davies_bouldin_score(X_cluster_sample, labels_opt)
    print(f"SpectralClustering (Optimized): Best Params={grid.best_params_}, Silhouette={sil_opt:.2f}, DB Index={db_opt:.2f}")
else:
    print("SpectralClustering (Optimized): Invalid clusters; scores not computable.")

# Save model
joblib.dump({'model': best_spectral, 'sample_size': n_sample, 'sample_idx': sample_idx if 'sample_idx' in locals() else None}, 'data/models/clustering/spectral_optimized.pkl')

# Basic Spectral for comparison (same params as basic)
spectral_basic = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42, n_neighbors=10)
labels_basic = spectral_basic.fit_predict(X_cluster_sample)

# Evaluate basic
sil_basic = silhouette_score(X_cluster_sample, labels_basic)
db_basic = davies_bouldin_score(X_cluster_sample, labels_basic)
print(f"SpectralClustering (Basic): Silhouette={sil_basic:.2f}, DB Index={db_basic:.2f}")

# Plot comparison (basic vs optimized)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Basic plot
scatter_basic = ax[0].scatter(df_sample['loc_pca1'], df_sample['loc_pca2'], c=labels_basic, cmap='viridis')
ax[0].set_title(f'Spectral Basic (n_clusters=3)\nSilhouette={sil_basic:.2f}, DB={db_basic:.2f}')
ax[0].set_xlabel('loc_pca1')
ax[0].set_ylabel('loc_pca2')

# Optimized plot
scatter_opt = ax[1].scatter(df_sample['loc_pca1'], df_sample['loc_pca2'], c=labels_opt, cmap='viridis')
ax[1].set_title(f'Spectral Optimized (n_clusters={grid.best_params_["n_clusters"]})\nSilhouette={sil_opt:.2f}, DB={db_opt:.2f}')
ax[1].set_xlabel('loc_pca1')
ax[1].set_ylabel('loc_pca2')

plt.tight_layout()
plt.show()