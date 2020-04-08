# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

from time import time

# ============================================================================

df = pd.read_csv('diabetes.csv', delimiter = ',')

X = df.drop('Outcome', axis = 1).values
y = df['Outcome'].values

# ============================================================================

pd.options.display.max_columns = 10
print(df.describe())
print()

# ----------------------------------------------------------------------------

plt.title("Number of Diabetes Cases")
sns.countplot(x = "Outcome", data = df)
plt.show()

# ----------------------------------------------------------------------------

sns.pairplot(df, hue = "Outcome", vars = df.columns[:-1])
plt.show()

# ----------------------------------------------------------------------------

plt.title("Heatmap of Variables Correlation")
sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn')
plt.show()

# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify = y)

# ----------------------------------------------------------------------------

neighbors = np.arange(1, 21)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
# ----------------------------------------------------------------------------
    
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# ----------------------------------------------------------------------------

print("Akurasi setiap K pada training set: ")
for i, accuracy in enumerate(train_accuracy):
    print(str(i + 1) + ": " + str(accuracy))
print()

# ----------------------------------------------------------------------------

print("Akurasi setiap K pada test set: ")
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

plt.title('Confusion Matrix pada Test Set')
sns.heatmap(confusion_matrix(y_pred, y_test), annot = True, annot_kws={"size": 16}, fmt='g')
plt.show()

# ============================================================================
