# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

import plotly.graph_objs as go
from plotly.offline import plot

import seaborn as sns

from time import time

# ============================================================================

df = pd.read_csv('Mall_Customers.csv', delimiter = ',')
X = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].values

# ============================================================================

pd.options.display.max_columns = 10
print(df.describe())
print()

# ----------------------------------------------------------------------------

sns.pairplot(df, vars = df.columns[2:])
plt.tight_layout()
plt.savefig('pairplot_mall_customers_asli.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------

plt.title("Heatmap Korelasi Antar Variabel Dataset asli")
ax = sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation = 68, ha="right")
plt.tight_layout()
plt.savefig('heatmap_mall_customers_asli.png', dpi=300)
plt.show()

# ============================================================================

kmax = 10

inertia = []
sil = []

for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters = k).fit(X)
    labels = kmeans.labels_
    
    inertia.append(kmeans.inertia_)
    sil.append(silhouette_score(X, labels, metric = 'euclidean'))

# ----------------------------------------------------------------------------
    
plt.title("Metode Elbow Dataset asli")
plt.plot(range(2, kmax + 1), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Error')
plt.tight_layout()
plt.savefig('elbow_mall_customers_asli.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------

plt.title("Sillhoutte Score Dataset asli")
plt.plot(range(2, kmax + 1), sil, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sillhoutte Score')
plt.tight_layout()
plt.savefig('siluet_mall_customers_asli.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------

print("Sillhoutte Score setiap K: ")
for i, score in enumerate(sil):
    print(str(i + 2) + ": " + str(score))
print()

# ----------------------------------------------------------------------------

highest_sil = max(sil)
k_highest_sil = np.argmax(sil) + 2

print("K terbaik adalah " + str(k_highest_sil) + " dengan Sillhoutte Score sebesar " + str(highest_sil))
print()

# ============================================================================

start_time = time()

kmeans = KMeans(n_clusters = k_highest_sil).fit(X)

print("--- Waktu yang dibutuhkan untuk melatih model adalah %s detik ---" % (time() - start_time))
print()

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# ----------------------------------------------------------------------------

df['label'] =  labels

trace1 = go.Scatter3d(
    x = df['Age'],
    y = df['Spending Score (1-100)'],
    z = df['Annual Income (k$)'],
    
    mode = 'markers',
    
    marker = dict(
        color = df['label'], 
        size = 15,
        line = dict(
            color = df['label'],
            width = 12
        ),
        opacity = 0.8
    )
)

data = [trace1]

layout = go.Layout(
    title= 'Clusters',
    scene = dict(
        xaxis = dict(title = 'Age'),
        yaxis = dict(title = 'Spending Score'),
        zaxis = dict(title = 'Annual Income')
    )
)

fig = go.Figure(data=data, layout=layout)
plot(fig)

# ============================================================================
