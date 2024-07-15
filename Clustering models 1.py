# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 00:31:28 2023

@author: SGvan
"""

import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load the dataset
file_path = 'standardized_customer_marketing_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'Dt_Customer' to datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%Y-%m-%d')

# Exclude non-numeric columns for outlier detection
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Check for outliers only in numeric columns
outliers = data[numeric_columns].apply(lambda x: (x < -3) | (x > 3))

# Cap the outliers at -3 and 3 for lower and upper bounds, respectively.
data_capped = data[numeric_columns].clip(lower=-3, upper=3)

# Replace the original numeric columns with the capped ones
data[numeric_columns] = data_capped

# Feature Engineering: Create new features
data['Customer_Tenure'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days
data['Total_Spending'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
data['Avg_Spending_Per_Product'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].mean(axis=1)
data['Engagement_Score'] = data[['Complain', 'NumWebVisitsMonth', 'Response']].mean(axis=1)
data['High_Spender'] = (data['Total_Spending'] > data['Total_Spending'].median()).astype(int)
data['High_Engagement'] = (data['Engagement_Score'] > data['Engagement_Score'].median()).astype(int)
data['Family_Size'] = data['Kidhome'] + data['Teenhome']
data['Is_Single'] = (data['Marital_Status_Single'] == 1).astype(int)

# Standardize the new features
scaler = StandardScaler()
new_features = ['Customer_Tenure', 'Total_Spending', 'Avg_Spending_Per_Product', 'Engagement_Score', 'Family_Size']
data[new_features] = scaler.fit_transform(data[new_features])

# Select a subset of features for clustering
features_to_use = ['Customer_Tenure', 'Total_Spending', 'Avg_Spending_Per_Product', 'Engagement_Score', 'Family_Size', 'Income', 'Recency']
X = data[features_to_use]

# Initialize clustering models
kmeans = KMeans(n_clusters=5, random_state=42)
dbscan = DBSCAN(eps=0.5, min_samples=5)
gmm = GaussianMixture(n_components=5, random_state=42)

# Fit the models
kmeans_labels = kmeans.fit_predict(X)
dbscan_labels = dbscan.fit_predict(X)
gmm_labels = gmm.fit_predict(X)

# Sampling the data for silhouette score calculation
sample_size = 500
X_sample = X.sample(n=sample_size, random_state=42)

# Recompute the labels for the sample data
kmeans_sample_labels = kmeans.predict(X_sample)
gmm_sample_labels = gmm.predict(X_sample)
dbscan_sample_labels = dbscan.fit_predict(X_sample)

# Calculate silhouette scores for the sample
kmeans_silhouette_sample = silhouette_score(X_sample, kmeans_sample_labels)
gmm_silhouette_sample = silhouette_score(X_sample, gmm_sample_labels)
dbscan_silhouette_sample = None
if len(set(dbscan_sample_labels)) > 1:
    dbscan_silhouette_sample = silhouette_score(X_sample[dbscan_sample_labels != -1], dbscan_sample_labels[dbscan_sample_labels != -1])

# Perform PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

# Perform t-SNE for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne_2d = tsne.fit_transform(X)

# Plot clusters after PCA
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans_labels)
plt.title('PCA - KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot clusters after t-SNE
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=kmeans_labels)
plt.title('t-SNE - KMeans Clusters')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()


# Plot clusters after PCA
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=dbscan_labels)
plt.title('PCA - DBSCAN Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot clusters after t-SNE
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=dbscan_labels)
plt.title('t-SNE - DBSCAN Clusters')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()

# Plot clusters after PCA
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=gmm_labels)
plt.title('PCA - gmm Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot clusters after t-SNE
plt.subplot(1, 2, 2)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=gmm_labels)
plt.title('t-SNE - gmm Clusters')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()

# Add the cluster labels to the data for the correlation matrix
data['Cluster_Labels'] = kmeans_labels

# Calculate the correlation matrix
clustered_corr = data[features_to_use + ['Cluster_Labels']].corr()

# Use seaborn's clustermap to reorder the correlation matrix and plot a heatmap
sns.clustermap(clustered_corr, annot=True, fmt=".2f", cmap="vlag", figsize=(10, 10))

plt.title('Heatmap of Clustered Correlation Matrix')
plt.show()

# Since DBSCAN may label some points as noise (-1), we need to handle this appropriately
# One approach is to exclude noise points for the heatmap visualization
if np.any(dbscan_labels != -1):
    data['DBSCAN_Cluster_Labels'] = dbscan_labels
    dbscan_corr = data[data['DBSCAN_Cluster_Labels'] != -1][features_to_use + ['DBSCAN_Cluster_Labels']].corr()
    sns.clustermap(dbscan_corr, annot=True, fmt=".2f", cmap="vlag", figsize=(10, 10))
    plt.title('Heatmap of Clustered Correlation Matrix for DBSCAN')
    plt.show()
else:
    print("DBSCAN did not form clusters, heatmap cannot be generated.")

# For GMM, we can directly use the labels to create the heatmap
data['GMM_Cluster_Labels'] = gmm_labels
gmm_corr = data[features_to_use + ['GMM_Cluster_Labels']].corr()
sns.clustermap(gmm_corr, annot=True, fmt=".2f", cmap="vlag", figsize=(10, 10))
plt.title('Heatmap of Clustered Correlation Matrix for GMM')
plt.show()

# Calculate Davies-Bouldin Index (the lower the better)
db_index_kmeans = davies_bouldin_score(X, kmeans_labels)
db_index_gmm = davies_bouldin_score(X, gmm_labels)

# Calculate Calinski-Harabasz Index (the higher the better)
ch_index_kmeans = calinski_harabasz_score(X, kmeans_labels)
ch_index_gmm = calinski_harabasz_score(X, gmm_labels)

# Calculate the proportion of noise points in the DBSCAN model
noise_proportion = np.sum(dbscan_labels == -1) / len(dbscan_labels)

print(f"Davies-Bouldin Index - KMeans: {db_index_kmeans}, GMM: {db_index_gmm}")
print(f"Calinski-Harabasz Index - KMeans: {ch_index_kmeans}, GMM: {ch_index_gmm}")
print(f"Proportion of noise points in DBSCAN: {noise_proportion}")