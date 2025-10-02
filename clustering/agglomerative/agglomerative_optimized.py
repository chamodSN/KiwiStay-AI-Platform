import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import numpy as np

np.random.seed(42)

# Load data
df = pd.read_csv('data/listings_discretized_enhanced.csv')
X_cluster = df[['log_price', 'loc_pca1', 'loc_pca2',
                'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']]

# Custom scorer


def sil_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return -1


param_grid = {'n_clusters': range(2, 11), 'linkage': [
    'ward', 'complete', 'average']}
grid = GridSearchCV(AgglomerativeClustering(),
                    param_grid, scoring=sil_scorer, cv=3)
grid.fit(X_cluster)
best_agg = grid.best_estimator_
labels_opt = best_agg.fit_predict(X_cluster)

sil_opt = silhouette_score(X_cluster, labels_opt)
db_opt = davies_bouldin_score(X_cluster, labels_opt)
print(
    f"Agglomerative (Optimized): Best Params={grid.best_params_}, Silhouette={sil_opt:.2f}, DB Index={db_opt:.2f}")

joblib.dump(best_agg, 'data/models/clustering/agglomerative_optimized.pkl')

# Plot comparison
agg_basic = AgglomerativeClustering(n_clusters=3)
labels_basic = agg_basic.fit_predict(X_cluster)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_basic)
ax[0].set_title('Agglomerative Basic')
ax[1].scatter(df['loc_pca1'], df['loc_pca2'], c=labels_opt)
ax[1].set_title('Agglomerative Optimized')
plt.show()
