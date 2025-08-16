import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time

mnist = fetch_openml('mnist_784', version=1)
#'data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[60000:], y.iloc[:60000], y.iloc[60000:]

digit = X.iloc[0].to_numpy()
# digit_image = digit.reshape(28, 28)
# plt.imshow(digit_image, cmap='binary')
# plt.show()

"""WYKRYWANIE JAKA CYFRA ZNAJDUJE SIĘ NA OBRAZKU"""

start = time.time()
print("start")

"""svm klasyfikuje binarnie, dlatego zostanie użyta strategia OvO (one vs one)
przez to wytrenowanych zostanie 45 klasyfikatorów, dlatego tak długo to zajmuje"""
# from sklearn.svm import SVC
# svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# print(svm_clf.predict([digit]))
# print(svm_clf.decision_function([digit]))


"""SGDClassifier"""
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix

# sgd_clf = SGDClassifier()
# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([digit]))
# print(sgd_clf.decision_function([digit]))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# conf_mx = confusion_matrix(y_train, y_train_pred)
# rows_sum = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx/rows_sum
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

"""NAPRAWA OBRAZKÓW CYFR"""
import random

index = random.randint(0, len(X_train))
noise = np.random.randint(510, 900, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(510, 900, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

clean_digit = X_train.values[index]
mess_digit = X_train_mod.values[index]

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
pred_digit = knn_clf.predict([mess_digit])


images = [clean_digit, mess_digit, pred_digit]
plt.figure(figsize=(9, 3))
for i, img in enumerate(images):
    plt.subplot(1, 3, i+1)
    plt.imshow(img.reshape(28, 28), cmap='binary')
    plt.axis('off')

plt.tight_layout()
plt.show()

end = time.time()
print(end - start)