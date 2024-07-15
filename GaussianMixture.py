from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'standardized_customer_marketing_data.csv'
data = pd.read_csv(file_path)

# Selecting the important features
important_features = ['CustomerTenure', 'EngagementScore']
X = data[important_features]

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_std)

# Applying Gaussian Mixture Models (GMM) with refined parameters
n_clusters_range = range(2, 11)  # Range of cluster numbers to try
best_gmm = None
best_silhouette_score = -1
best_davies_bouldin_score = np.inf
best_n_clusters = 0

for n_clusters in n_clusters_range:
    # Trying different covariance types and increasing n_init to improve initialization
    for covariance_type in ['spherical', 'diag', 'tied', 'full']:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                              n_init=10, random_state=42)
        gmm_labels = gmm.fit_predict(X_std)

        # Calculating the silhouette and Davies-Bouldin scores for each model
        if len(set(gmm_labels)) > 1:  # Avoid calculating score for 1 cluster
            silhouette_avg = silhouette_score(X_std, gmm_labels)
            davies_bouldin_avg = davies_bouldin_score(X_std, gmm_labels)

            # Check if this model is the best so far
            if silhouette_avg > best_silhouette_score and davies_bouldin_avg < best_davies_bouldin_score:
                best_silhouette_score = silhouette_avg
                best_davies_bouldin_score = davies_bouldin_avg
                best_gmm = gmm
                best_n_clusters = n_clusters
                best_covariance_type = covariance_type

# Visualization of the best clustering result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_gmm.predict(X_std), cmap='viridis')
plt.title(f'GMM Clustering with t-SNE visualization\nBest model: {best_n_clusters} clusters, {best_covariance_type} covariance')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar()
plt.show()

# Print the best model's details
print(f"Best number of clusters: {best_n_clusters}")
print(f"Best silhouette score: {best_silhouette_score}")
print(f"Best Davies-Bouldin score: {best_davies_bouldin_score}")
print(f"Best covariance type: {best_covariance_type}")

# Save the best model
import joblib
joblib.dump(best_gmm, 'best_gmm_model.joblib')

print("The best GMM model has been saved.")
