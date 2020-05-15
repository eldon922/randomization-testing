# -*- coding: utf-8 -*-

# Dua baris dibawah adalah kode untuk menghitung ADJUSTED RAND INDEX
# variabel labels_asli dan labels_proyeksi perlu diambil dari hasil
# clustering yang sudah dilakukan. Tips: dapat menggunakan fitur copy/paste
# variabel pada perangkat lunak Spyder di tab Variable explorer

# from sklearn.metrics import adjusted_rand_score
# adjusted_rand_score(labels_asli, labels_proyeksi)


# Berikut adalah kode untuk menguji hasil Random Projection Perturbation
# apakah berada pada rentang jarak Euclidean yang benar
import pandas as pd

from scipy.spatial import distance
import sys

df_asli = pd.read_csv('train.csv', delimiter = ',')
df_projected = pd.read_csv('projected_train.csv', delimiter = ',')
asli = df_asli.drop(['subject', 'Activity'], axis = 1).values
projected = df_projected.drop(['subject', 'Activity'], axis = 1).values

result = []
eps = 0.52
for i in range(0, asli.shape[0]):
    for j in range(i, asli.shape[0]):
        d_asli = distance.euclidean(asli[i], asli[j])
        d_projected = distance.euclidean(projected[i], projected[j])
        if (1-eps) * d_asli**2 > d_projected**2 or d_projected**2 > (1+eps) * d_asli**2:
            print("------------ERROR: " + str(i) + " " + str(j) + " ------------------")
            sys.exit()
        else:
            print(str(i) + " " + str(j) + ": " + str(d_asli) + " " + str(d_projected))
            
print("ok")
            