import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import time

from scipy.ndimage import shift
def shift_image(image, dx, dy):
    image = image.reshape(28, 28)
    shifted = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted.reshape(784)

def more_exaples(X, y):
    left = np.apply_along_axis(shift_image, 1, X, dx=-1, dy=0)
    right = np.apply_along_axis(shift_image, 1, X, dx=1, dy=0)
    up = np.apply_along_axis(shift_image, 1, X, dx=0, dy=-1)
    down = np.apply_along_axis(shift_image, 1, X, dx=0, dy=1)

    # glob = np.concatenate([X, left, right, up, down], axis=0)
    # yy = np.concatenate([y, y, y, y, y], axis=0)

    # fig, axes = plt.subplots(4, 4, figsize=(8, 4))
    # for i, ax in enumerate(axes.flat):
    #     if i == 15: break
    #     ax.imshow(glob[i].reshape(28, 28), cmap='binary')
    #     ax.axis('off')
    #     ax.set_title(yy[i])
    # plt.show()
    # exit()


    return np.concatenate([X, left, right, up, down], axis=0), np.concatenate([y, y, y, y, y], axis=0)

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

# digit = X.iloc[0].to_numpy()
# digit = shift_image(digit, 5, 5)
# plt.imshow(digit.reshape(28, 28), cmap='binary')
# plt.show()
# exit()

#X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[60000:], y.iloc[:60000], y.iloc[60000:]
new_X, new_y = X.iloc[:].to_numpy(), y.iloc[:].to_numpy()
new_X, new_y = more_exaples(new_X, new_y)

print(new_X.shape, new_y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, train_size=0.9)

# fig, axes = plt.subplots(4, 4, figsize=(8, 4))
# for i, ax in enumerate(axes.flat):
#     if i == 15: break
#     ax.imshow(X_train[i].reshape(28, 28), cmap='binary')
#     ax.axis('off')
#     ax.set_title(y_train[i])
# plt.show()


"""WYKRYWANIE JAKA CYFRA ZNAJDUJE SIĘ NA OBRAZKU"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

"""KNeighborsClassifier"""

knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance', p=1)
knn_clf.fit(X_train_scaled, y_train)

final_preds = knn_clf.predict(X_test_scaled)
print(accuracy_score(y_test, final_preds))

# best score: 0.9753142857142857






















# Best: 0.95615
# {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# parameters = {'n_neighbors': [5, 10],
#               'weights': ['uniform', 'distance'],
#               'p': [1, 2]
#               }

# grid_search = GridSearchCV(knn_clf, parameters, cv=3,
#                            scoring='accuracy',
#                            return_train_score=True)

# start = time.time()
# print("start KNeighborsClassifier")

# grid_search.fit(X_train_scaled, y_train)

# end = time.time()
# print("finish, total time (seconds): ", end - start)


# cvres = grid_search.cv_results_
# for score, params in zip(cvres['mean_test_score'], cvres['params']):
#     print(f"Accuracy: {score:.3f}", params)

# # najlepszy:
# print("Best:", grid_search.best_score_)
# print(grid_search.best_params_)