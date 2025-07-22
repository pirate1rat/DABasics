import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def load_data(path: str):
    return pd.read_csv(path)

def basic_info(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.shape)
    print(df.columns)

def graphs(df: pd.DataFrame):
    corr_matrix = df.corr()
    print(corr_matrix['Survived'].sort_values(ascending=False))
    print(df.groupby('Pclass')['Cabin'].sum())

    # attrbs = ['Survived', 'Pclass', 'Sex', 'encoded']
    # scatter_matrix(df[attrbs], figsize=(7, 7))

    # counts = df['Survived'].value_counts()
    # labels = ['Nie przeżył', 'Przeżył']
    # plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    # plt.title('Przeżywalność')
    # plt.show()

    df['Age'].hist(bins=20)
    plt.show()

df = load_data("titanic_project\\titanic_v2.csv")
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
         'Cabin'
         ]]

#print(df['Sex'].value_counts())
split = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
for train_idx, test_idx in split.split(df, df['Sex']):
    train_set = df.loc[train_idx]
    test_set = df.loc[test_idx]

labels = train_set['Survived']
train_set = train_set.drop(columns=['Survived'])

#sprawdzenie czy dobrze rozlozylo
#print(train_set['Sex'].value_counts() / len(train_set))


#train_set['Sex'] = train_set['Sex'].map({'female': 0, 'male': 1})
#train_set['Cabin'] = np.where(train_set['Cabin'].isna(), 0, 1)
#graphs(train_set)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class CabinSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keep_cabins=True):
        self.keep_cabins = keep_cabins
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        if 'Cabin' in X_.columns:
            if self.keep_cabins:
                X_['Cabin'] = X_['Cabin'].notna().astype(int)
            else:
                X_ = X_.drop(columns='Cabin')
        return X_

def sex_transform(X):
    return X.map({'female': 0, 'male': 1}).values.reshape(-1, 1)

col_pipeline = ColumnTransformer([
    ('sex', FunctionTransformer(sex_transform, feature_names_out='one-to-one'), 'Sex'),
    ('std_scaller', StandardScaler(), ['Age']),
], remainder='passthrough')

full_pipeline = Pipeline([
    ('cabin', CabinSelector(False)),
    ('col', col_pipeline),
    ('imputer', SimpleImputer(strategy='median')),
])

prepared_df = full_pipeline.fit_transform(train_set)






from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion':        ['gini', 'entropy'],       # funkcja oceny czystości
    'max_depth':        [None, 5, 10, 20, 30],      # ograniczenie głębokości drzewa
    'min_samples_split':[2, 5, 10],                 # min. próbek do podziału w węźle
    'min_samples_leaf': [1, 2, 4],                  # min. próbek w liściu
    'max_features':     [None, 'sqrt', 'log2'],     # liczba cech losowana przy podziale
}
tree_reg = DecisionTreeClassifier()


print("rozpoczynam nauke\n")
grid_search = GridSearchCV(tree_reg, param_grid, cv=10,
                           scoring='accuracy',
                           return_train_score=True)
grid_search.fit(prepared_df, labels)
print("koniec nauki")


print(grid_search.best_params_)
print(grid_search.best_estimator_)
# cvres = grid_search.cv_results_
# for score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(f"Accuracy: {score:.3f}", params)

final_model = grid_search.best_estimator_
X_test = test_set.drop('Survived', axis=1)
y_test = test_set['Survived'].copy()

from sklearn.metrics import accuracy_score, classification_report
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
print("Test accuracy:", accuracy_score(y_test, final_predictions))
print(classification_report(y_test, final_predictions))