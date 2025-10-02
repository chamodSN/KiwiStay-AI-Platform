import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import numpy as np

np.random.seed(42)

# Load data (use numerical/transformed features for clustering: price, loc_pca1/2, availability_365, etc.)
df = pd.read_csv('data/listings_discretized_enhanced.csv')  # Use final preprocessed
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Basic DBSCAN (default params)
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_cluster)

# Evaluate (handle if only one cluster or noise)
if len(set(labels)) > 1:
    sil = silhouette_score(X_cluster, labels)
    db = davies_bouldin_score(X_cluster, labels)
    print(f"DBSCAN (Basic): Silhouette={sil:.2f}, DB Index={db:.2f}")
else:
    print("DBSCAN (Basic): Only one cluster or noise; scores not computable.")

# Label segments based on cluster means (ignore noise -1)
valid_labels = labels[labels != -1]
valid_df = df[labels != -1]
if len(set(valid_labels)) > 0:
    cluster_means = valid_df.groupby(valid_labels)['log_price'].mean()
    segment_labels = {i: f'segment_{i}' for i in set(valid_labels)}  # Customize based on means, e.g., 'luxury' for high price
    print("Cluster Means (Price):", cluster_means)

# Save model
os.makedirs('data/models/clustering', exist_ok=True)
joblib.dump(dbscan, 'data/models/clustering/dbscan_basic.pkl')

# Plot clusters (using loc_pca for viz; noise as gray)
plt.scatter(df['loc_pca1'], df['loc_pca2'], c=labels, cmap='viridis')
plt.title('DBSCAN Clusters')
plt.show()
