import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
#'data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

"""mini kontrola danych
print(X.shape, y.shape)
print(y.iloc[0])
digit = X.iloc[0].to_numpy()
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap='binary')
plt.axis('off')
plt.show()"""

#wystarczy tyle bo podobno zestaw już jest potasowany
X_train, X_test, y_train, y_test = X.iloc[:60000], X.iloc[60000:], y.iloc[:60000], y.iloc[60000:]


"""UPROSZCZONY PROBLEM - KLASYFIKATOR BINARNY"""
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
#sgd_clf.fit(X_train, y_train_5)

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# print(scores)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# from sklearn.metrics import confusion_matrix, f1_score
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))
# print("precision score: ", precision_score(y_train_5, y_train_pred))
# print("recall score: ", recall_score(y_train_5, y_train_pred))
# print("f1 socre: ", f1_score(y_train_5, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precision, recalls, threshold = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_threshold(precision, recalls, threshold):
    plt.plot(threshold, precision[:-1], 'b--', label='Precyzja')
    plt.plot(threshold, recalls[:-1], 'g-', label='Pełność')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Threshold")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_precision_vs_recall(precision, recalls):
    plt.figure()
    plt.plot(recalls, precision, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision w funkcji Recall')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# plot_precision_recall_threshold(precision, recalls, threshold)
# plot_precision_vs_recall(precision, recalls)

"""ustawianie dowolnej precyzji
threshold_90_precision = threshold[np.argmax(precision >= 0.9)]
print("threshold_90_precision: ", threshold_90_precision)
y_train_90 = (y_scores > threshold_90_precision)
print("precision score: ", precision_score(y_train_5, y_train_90))
print("recall score: ", recall_score(y_train_5, y_train_90))
"""

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Odsetek fałszywie pozytywnych")
    plt.ylabel("Odsetek prawdziwie pozytywnych")
    plt.title("ROC - receiver operating characteristic")
    plt.legend(loc='best')

    plt.show()

#plot_roc_curve(fpr, tpr)
print(roc_auc_score(y_train_5, y_scores))


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]
y_pred_forest = (y_scores_forest >= 0.5).astype(int)
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

print(roc_auc_score(y_train_5, y_scores_forest))
print("precision score: ", precision_score(y_train_5, y_pred_forest))
print("recall score: ", recall_score(y_train_5, y_pred_forest))

plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Las losowy')