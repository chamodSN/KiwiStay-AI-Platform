
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import numpy as np

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized_enhanced.csv')
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Custom scorer for GridSearch (max silhouette)
def sil_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    unique_labels = len(set(labels))
    if unique_labels > 1 and unique_labels < len(X):  # Avoid all noise or one cluster
        return silhouette_score(X, labels)
    return -1

param_grid = {'eps': [0.1, 0.5, 1.0, 1.5], 'min_samples': [3, 5, 10, 15]}
grid = GridSearchCV(DBSCAN(), param_grid, scoring=sil_scorer, cv=3)
grid.fit(X_cluster)
best_dbscan = grid.best_estimator_
labels_opt = best_dbscan.fit_predict(X_cluster)

# Evaluate
unique_opt = len(set(labels_opt))
if unique_opt > 1 and unique_opt < len(X_cluster):
    sil_opt = silhouette_score(X_cluster, labels_opt)
    db_opt = davies_bouldin_score(X_cluster, labels_opt)
    print(f"DBSCAN (Optimized): Best Params={grid.best_params_}, Silhouette={sil_opt:.2f}, DB Index={db_opt:.2f}")
else:
    print("DBSCAN (Optimized): Invalid clusters; scores not computable.")

joblib.dump(best_dbscan, 'data/models/clustering/dbscan_optimized.pkl')

# Plot comparison (basic vs opt)
dbscan_basic = DBSCAN()
labels_basic = dbscan_basic.fit_predict(X_cluster)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_basic, cmap='viridis')
ax[0].set_title('DBSCAN Basic')
ax[1].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_opt, cmap='viridis')
ax[1].set_title('DBSCAN Optimized')
plt.show()

