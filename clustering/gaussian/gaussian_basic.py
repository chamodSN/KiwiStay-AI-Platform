import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import numpy as np

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized.csv')
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Basic GaussianMixture (default n_components=3)
gm = GaussianMixture(n_components=3, random_state=42)
labels = gm.fit_predict(X_cluster)

# Evaluate
sil = silhouette_score(X_cluster, labels)
db = davies_bouldin_score(X_cluster, labels)
print(f"GaussianMixture (Basic): Silhouette={sil:.2f}, DB Index={db:.2f}")

# Label segments based on cluster means
cluster_means = df.groupby(labels)['log_price'].mean()
segment_labels = {0: 'budget urban', 1: 'luxury tourist', 2: 'mid-range rural'}
print("Cluster Means (Price):", cluster_means)

# Save model
os.makedirs('data/models/clustering', exist_ok=True)
joblib.dump(gm, 'data/models/clustering/gaussian_basic.pkl')

# Plot clusters
plt.scatter(df['loc_pca1'], df['loc_pca2'], c=labels)
plt.title('GaussianMixture Clusters')
plt.show()
