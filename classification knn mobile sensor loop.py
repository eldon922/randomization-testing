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

from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection

class CustomGaussianRandomProjection(GaussianRandomProjection):
    def __init__(self, n_components, eps, randomProjectionMatrix):
        super().__init__(n_components=n_components, eps=eps)
        self._randomProjectionMatrix = randomProjectionMatrix

    def _make_random_matrix(self, n_components, n_features):
        # TODO kasih if?
        return self._randomProjectionMatrix

# ============================================================================

df_train = pd.read_csv('train.csv', delimiter = ',')

X_train_df = df_train.drop(['subject', 'Activity'],axis = 1).values
label_np = df_train['Activity'].values

# ----------------------------------------------------------------------------

label_np = label_np.ravel()
le = LabelEncoder()

y_train = le.fit_transform(label_np)

# ============================================================================

df_test = pd.read_csv('test.csv', delimiter = ',')

X_test_df = df_test.drop(['subject', 'Activity'], axis = 1).values
label_np_test = df_test['Activity'].values

# ----------------------------------------------------------------------------

label_np_test = label_np_test.ravel()
y_test = le.fit_transform(label_np_test)

# ============================================================================

df_train_test = df_train.append(df_test)

# ----------------------------------------------------------------------------

# =============================================================================
# print(df_train_test.describe())
# print()
# =============================================================================

# ----------------------------------------------------------------------------

# =============================================================================
# plt.title("Activities Count")
# ax = sns.countplot(x = "Activity", data = df_train_test)
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 20, ha="right")
# plt.show()
# =============================================================================

# ============================================================================
y_pred = []
for j in range(2, 21):
    creator = GaussianRandomProjection(n_components=j)
    X = creator._make_random_matrix(j, X_train_df.shape[1])
    
    transformer = CustomGaussianRandomProjection(j, 0.1, X)
    X_train = transformer.fit_transform(X_train_df)
    X_test = transformer.fit_transform(X_test_df)
    
    k_min = 15
    neighbors = np.arange(k_min,25)
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
        
    plt.title(str(j) + ': k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    # ----------------------------------------------------------------------------
    
    # =============================================================================
    # print("Akurasi setiap K pada training set: ")
    # for i, accuracy in enumerate(train_accuracy):
    #     print(str(i + k_min) + ": " + str(accuracy))
    # print()
    # 
    # # ----------------------------------------------------------------------------
    # 
    # print("Akurasi setiap K pada test set: ")
    # for i, accuracy in enumerate(test_accuracy):
    #     print(str(i + k_min) + ": " + str(accuracy))
    # print()
    # =============================================================================
    
    # ----------------------------------------------------------------------------
    
    highest_test_accuracy = test_accuracy.max()
    k_highest_test_accuracy = np.argmax(test_accuracy) + k_min
    
    print(str(j) + ": K terbaik adalah " + str(k_highest_test_accuracy) + " dengan akurasi test set sebesar " + str(highest_test_accuracy))
    print()
    
    # ============================================================================
    
    start_time = time()
    
    knn = KNeighborsClassifier(n_neighbors = k_highest_test_accuracy)
    knn.fit(X_train, y_train)
    
    # print("--- Waktu yang dibutuhkan untuk melatih model adalah %s detik ---" % (time() - start_time))
    # print()
    
    # print("Akurasi pada model KNN yang digunakan: " + str(knn.score(X_test, y_test)))
    # print()
    
    # ----------------------------------------------------------------------------
    
    start_time = time()
    
    y_pred.append(knn.predict(X_test))
    
    # print("--- Waktu yang dibutuhkan untuk melakukan prediksi adalah %s detik ---" % (time() - start_time))
    # print()
    
    # =============================================================================
    # plt.title('Confusion Matrix pada Test Set')
    # sns.heatmap(confusion_matrix(y_pred, y_test), annot = True, annot_kws={"size": 16}, fmt='g')
    # plt.show()
    # =============================================================================
    
    # ============================================================================
