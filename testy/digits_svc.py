import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

#XGBoost/LightGBM/CatBoost
mnist = fetch_openml('mnist_784', version=1)
X = mnist['data']
y = mnist['target']

X = X[:10000]
y = y[:10000]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_test = ss.fit_transform(X_test)
X_train = ss.fit_transform(X_train)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, reciprocal

svc = SVC(C=3.8786881587000437, gamma=0.0017076019229344522, decision_function_shape='ovo')
#svc = SVC(gamma= 0.016599452033620267, C=2.1560186404424364, decision_function_shape='ovr') 
#ovr_clf = OneVsRestClassifier(svc)
# params = {'estimator__gamma': uniform(0.001, 0.1), 'estimator__C': uniform(2, 1)}
# rand_search = RandomizedSearchCV(ovr_clf, params, n_iter=5, cv=3, verbose=3, random_state=42)
# rand_search.fit(X_train, y_train)

# print(rand_search.best_estimator_)
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, svc.predict(X_test)))