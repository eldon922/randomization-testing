# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

from time import time

# ============================================================================

df = pd.read_csv('train.csv', delimiter = ',')
X_df = df.drop(['subject', 'Activity'], axis = 1).values

# ----------------------------------------------------------------------------
mylabels=[]
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
for i in range(2, 21):
    transformer = GaussianRandomProjection(n_components=i)
    X = transformer.fit_transform(X_df)
    
    pca = PCA(n_components = 2)
    X = pca.fit_transform(X)
    
# # ============================================================================

# print(df.describe())
# print()

# # ----------------------------------------------------------------------------

# plt.title("Activities Count")
# ax = sns.countplot(x = "Activity", data = df)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 20, ha="right")
# plt.show()

# # ----------------------------------------------------------------------------

    df_pca = pd.DataFrame(X, columns = ['PCA 1', 'PCA 2']).join(df[["Activity"]])

# plt.title("Data Plot")
# sns.scatterplot(x = 'PCA 1', y = 'PCA 2', hue = "Activity", data = df_pca)
# plt.legend(bbox_to_anchor=(1, 1))
# plt.show()

# # ============================================================================
    
    kmax = 10
    
    inertia = []
    sil = []
    
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters = k).fit(X)
        labels = kmeans.labels_
        
        inertia.append(kmeans.inertia_)
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    
# # ----------------------------------------------------------------------------
    
# plt.title("Elbow Method")
# plt.plot(range(2, kmax + 1), inertia, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

# # ----------------------------------------------------------------------------

# plt.title("Sillhoutte Score")
# plt.plot(range(2, kmax + 1), sil, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sillhoutte Score')
# plt.show()

# # ----------------------------------------------------------------------------

# print("Sillhoutte Score setiap K: ")
# for i, score in enumerate(sil):
#     print(str(i + 2) + ": " + str(score))
# print()

# # ----------------------------------------------------------------------------
    
    highest_sil = max(sil)
    k_highest_sil = np.argmax(sil) + 2
    
    print(str(i) + ": K terbaik adalah " + str(k_highest_sil) + " dengan Sillhoutte Score sebesar " + str(highest_sil))
    print()
    
    # ============================================================================
    
    start_time = time()
    
    kmeans = KMeans(n_clusters = k_highest_sil).fit(X)
    
    # print("--- Waktu yang dibutuhkan untuk melatih model adalah %s detik ---" % (time() - start_time))
    # print()
    
    mylabels.append(kmeans.labels_)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # ----------------------------------------------------------------------------

    df_pca['Cluster'] = labels
    df_pca['Cluster'] = df_pca['Cluster'].map({0: 'Moving', 1: 'Not Moving'})
    
    plt.title(str(i) + ": Cluster Results")
    sns.scatterplot(x = "PCA 1", y = "PCA 2", hue = "Cluster", data = df_pca)
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1],
        s=250, 
        marker='*',
        c='red', 
        edgecolor='black'
    )
    plt.show()

# ============================================================================
