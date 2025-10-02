import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

np.random.seed(42)

# Load data (use numerical/transformed features for clustering: price, loc_pca1/2, availability_365, etc.)
df = pd.read_csv('data/listings_discretized_enhanced.csv')  # Use final preprocessed
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Basic KMeans (default k=3 for segments)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_cluster)

# Evaluate
sil = silhouette_score(X_cluster, labels)
db = davies_bouldin_score(X_cluster, labels)
print(f"KMeans (Basic): Silhouette={sil:.2f}, DB Index={db:.2f}")

# Label segments based on cluster means
cluster_means = df.groupby(labels)['log_price'].mean()  # Example; extend to other features
segment_labels = {0: 'budget urban', 1: 'luxury tourist', 2: 'mid-range rural'}  # Based on means
print("Cluster Means (Price):", cluster_means)

# Save model
os.makedirs('data/models/clustering', exist_ok=True)
joblib.dump(kmeans, 'data/models/clustering/kmeans_basic.pkl')

# Plot clusters (using loc_pca for viz)
plt.scatter(df['loc_pca1'], df['loc_pca2'], c=labels)
plt.title('KMeans Clusters')
plt.show()