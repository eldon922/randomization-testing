# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

from time import time

# ============================================================================

df_train = pd.read_csv('projected2_train.csv', delimiter = ',')

X_train = df_train.drop(['subject', 'Activity'],axis = 1).values
label_np = df_train['Activity'].values

# X = np.random.rand(X_train.shape[1],X_train.shape[1])
# X_train = X_train @ X

# ----------------------------------------------------------------------------

label_np = label_np.ravel()
le = LabelEncoder()

y_train = le.fit_transform(label_np)

# ============================================================================

df_test = pd.read_csv('projected2_test.csv', delimiter = ',')

X_test = df_test.drop(['subject', 'Activity'], axis = 1).values
label_np_test = df_test['Activity'].values

# X_test = X_test @ X

# ----------------------------------------------------------------------------

label_np_test = label_np_test.ravel()
y_test = le.fit_transform(label_np_test)

# ============================================================================

df_train_test = df_train.append(df_test)

# ----------------------------------------------------------------------------

pd.options.display.max_columns = 5
print(df_train_test.drop(['subject'],axis = 1).describe())
print()

# ----------------------------------------------------------------------------

plt.title("Distribusi Label (Aktivitas)")
ax = sns.countplot(x = "Activity", data = df_train_test)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 20, ha="right")
plt.tight_layout()
plt.savefig('distribusi_label_mobile_sensor_asli.png', dpi=300)
plt.show()

# ============================================================================

neighbors = np.arange(1,31)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
# ----------------------------------------------------------------------------
    
plt.title('Dataset Asli')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('akurasi_mobile_sensor_asli.png', dpi=300)
plt.show()

# ----------------------------------------------------------------------------

print("Akurasi setiap K pada training set dataset asli: ")
for i, accuracy in enumerate(train_accuracy):
    print(str(i + 1) + ": " + str(accuracy))
print()

# ----------------------------------------------------------------------------

print("Akurasi setiap K pada test set dataset asli: ")
for i, accuracy in enumerate(test_accuracy):
    print(str(i + 1) + ": " + str(accuracy))
print()

# ----------------------------------------------------------------------------

highest_test_accuracy = test_accuracy.max()
k_highest_test_accuracy = np.argmax(test_accuracy) + 1

print("K terbaik adalah " + str(k_highest_test_accuracy) + " dengan akurasi test set sebesar " + str(highest_test_accuracy))
print()

# ============================================================================

start_time = time()

knn = KNeighborsClassifier(n_neighbors = k_highest_test_accuracy)
knn.fit(X_train, y_train)

print("--- Waktu yang dibutuhkan untuk melatih model adalah %s detik ---" % (time() - start_time))
print()

print("Akurasi pada model KNN yang digunakan: " + str(knn.score(X_test, y_test)))
print()

# ----------------------------------------------------------------------------

start_time = time()

y_pred = knn.predict(X_test)

print("--- Waktu yang dibutuhkan untuk melakukan prediksi adalah %s detik ---" % (time() - start_time))
print()

plt.title('Dataset Asli')
sns.heatmap(confusion_matrix(y_pred, y_test), annot = True, annot_kws={"size": 16}, fmt='g')
plt.tight_layout()
plt.savefig('confusion_matrix_mobile_sensor_asli.png', dpi=300)
plt.show()

# ============================================================================
