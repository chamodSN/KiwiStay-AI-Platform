import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized_enhanced.csv')
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Custom scorer for GridSearch (max silhouette)
def sil_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return -1

# GridSearch for n_clusters
param_grid = {'n_clusters': range(2, 11)}
grid = GridSearchCV(KMeans(random_state=42), param_grid, scoring=sil_scorer, cv=3)
grid.fit(X_cluster)
best_kmeans = grid.best_estimator_
labels_opt = best_kmeans.fit_predict(X_cluster)

# Evaluate optimized model
sil_opt = silhouette_score(X_cluster, labels_opt)
db_opt = davies_bouldin_score(X_cluster, labels_opt)
print(f"KMeans (Optimized): Best Params={grid.best_params_}, Silhouette={sil_opt:.2f}, DB Index={db_opt:.2f}")

# Save optimized model
os.makedirs('data/models/clustering', exist_ok=True)
joblib.dump(best_kmeans, 'data/models/clustering/kmeans_optimized.pkl')

# Basic KMeans for comparison
kmeans_basic = KMeans(n_clusters=3, random_state=42)
labels_basic = kmeans_basic.fit_predict(X_cluster)

# Evaluate basic model
sil_basic = silhouette_score(X_cluster, labels_basic)
db_basic = davies_bouldin_score(X_cluster, labels_basic)
print(f"KMeans (Basic): Silhouette={sil_basic:.2f}, DB Index={db_basic:.2f}")

# Plot comparison (basic vs optimized)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Basic KMeans plot
ax[0].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_basic, cmap='viridis')
ax[0].set_title(f'KMeans Basic (n_clusters=3)\nSilhouette={sil_basic:.2f}, DB={db_basic:.2f}')
ax[0].set_xlabel('loc_pca1')
ax[0].set_ylabel('loc_pca2')

# Optimized KMeans plot
ax[1].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_opt, cmap='viridis')
ax[1].set_title(f'KMeans Optimized (n_clusters={grid.best_params_["n_clusters"]})\nSilhouette={sil_opt:.2f}, DB={db_opt:.2f}')
ax[1].set_xlabel('loc_pca1')
ax[1].set_ylabel('loc_pca2')

plt.tight_layout()
plt.show()