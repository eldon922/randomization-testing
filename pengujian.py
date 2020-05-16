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

eps = 0.52
for i in range(0, asli.shape[0]):
    for j in range(i, asli.shape[0]):
        d_asli = distance.euclidean(asli[i], asli[j])
        d_projected = distance.euclidean(projected[i], projected[j])
        if (1-eps) * d_asli**2 > d_projected**2 or d_projected**2 > (1+eps) * d_asli**2:
            print("------------ERROR: " + str(i+1) + " " + str(j+1) + ": " + 
                  str(d_asli) + " " + str(d_projected) + " ------------------")
            sys.exit()
        else:
            print(str(i+1) + " " + str(j+1) + ": " + str(d_asli) + " " + str(d_projected))
            
print("ok")



# Berikut adalah kode untuk menguji hasil Random Rotation Perturbation
# apakah mempunyai jarak Euclidean yang sama persis dengan aslinya
df_asli = pd.read_csv('Mall_Customers.csv', delimiter = ',')
asli = df_asli[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].values
df_rotated = pd.read_csv('rotated_Mall_Customers.csv', delimiter = ',')
rotated = df_rotated[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].values

for i in range(0, asli.shape[0]):
    for j in range(i, asli.shape[0]):
        d_asli = distance.euclidean(asli[i], asli[j])
        d_rotated = distance.euclidean(rotated[i], rotated[j])
        if round(d_asli, 2) != round(d_rotated, 2) and round(d_asli, 3) != round(d_rotated, 3):
            print("------------ERROR: " + str(i+1) + " " + str(j+1) + ": " + 
                  str(d_asli) + " " + str(d_rotated) + " ------------------")
            sys.exit()
        else:
            print(str(i+1) + " " + str(j+1) + ": " + str(d_asli) + " " + str(d_rotated))
            
print("ok")
