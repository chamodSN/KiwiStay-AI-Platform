import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized.csv')  # Use enhanced for consistency
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# FIX: Downsample to avoid memory error (adjust n_sample as needed; 2000 fits ~16GB RAM)
n_sample = 2000  # Safe starting point; increase to 5000 if more RAM available
if len(X_cluster) > n_sample:
    sample_idx = np.random.choice(len(X_cluster), n_sample, replace=False)
    X_cluster_sample = X_cluster.iloc[sample_idx]
    df_sample = df.iloc[sample_idx]
    print(f"Downsampled to {n_sample} samples for memory efficiency.")
else:
    X_cluster_sample = X_cluster
    df_sample = df

# Basic SpectralClustering (n_clusters=3; use nearest_neighbors affinity to sparsify matrix)
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42, n_neighbors=10)  # FIX: Sparsify to reduce memory
labels = spectral.fit_predict(X_cluster_sample)

# Evaluate (only if >1 cluster)
unique_labels = len(set(labels))
if unique_labels > 1:
    sil = silhouette_score(X_cluster_sample, labels)
    db = davies_bouldin_score(X_cluster_sample, labels)
    print(f"SpectralClustering (Basic): Silhouette={sil:.2f}, DB Index={db:.2f}")
else:
    print("SpectralClustering (Basic): Only one cluster detected; scores not computable.")

# Label segments based on cluster means
cluster_means = df_sample.groupby(labels)['log_price'].mean()
segment_labels = {i: f'Segment {i} (Price: ${cluster_means.get(i, 0):.0f})' for i in set(labels)}
print("Cluster Means (Price):", cluster_means)

# Save model (with sample info)
os.makedirs('data/models/clustering', exist_ok=True)
joblib.dump({'model': spectral, 'sample_size': n_sample, 'sample_idx': sample_idx if 'sample_idx' in locals() else None}, 'data/models/clustering/spectral_basic.pkl')

# Plot clusters (using loc_pca for viz)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_sample['loc_pca1'], df_sample['loc_pca2'], c=labels, cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.title(f'SpectralClustering Clusters (n={n_sample} samples)')
plt.xlabel('loc_pca1')
plt.ylabel('loc_pca2')
plt.show()