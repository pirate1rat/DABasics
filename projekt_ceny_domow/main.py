import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Niestandardowy Transformator
# musi mieć fit, transform i fit_transform
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): #zadnych zmiennych *args ani **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Pokoje_na_rodzine = X[:, rooms_ix] / X[:, households_ix]
        Populacja_na_rodzine = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            Sypialnie_na_pokoje = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, Pokoje_na_rodzine, Populacja_na_rodzine, Sypialnie_na_pokoje]
        else:
            return np.c_[X, Pokoje_na_rodzine, Populacja_na_rodzine] # np.c_ np.column_stack    



"""ALTERNATYWNE ROZKLADANIE DANYCH NA ZESTAWY
#mechanizm do tworzenia stalych zestawow danych pomimo dokladania nowych
#odkladamy X% najmniejszych hashy jako zb. testowy
#nawet jesli dolozymy nowe dane, to tamte stare i tak trafia gdzie byly wczesniej

#w tym przypadku housing nie ma kolumny z identyfikatorami, trzeba dodac taka linijke
# housing_with_id = housing_df.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.3, "index")
def test_set_check(identifier, test_ratio: float):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio* 2**32

def split_train_test_by_id(data: pd.DataFrame, test_ratio: float, id_column: str):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
################
    
INNY SPOSOB:
def split_train_test(data: pd.DataFrame, test_ratio: float):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

"""

"""WCZYTYWANIE DANYCH""" 
def load_data():
    return pd.read_csv("projekt_ceny_domow\\housing.csv")


housing_df = load_data()
np.random.seed(0)

# print(housing_df.info())
# print(housing_df.describe())
# print(housing_df['ocean_proximity'].unique()) #value_counts()
# housing_df.hist(bins=50, figsize=(12, 7))
# plt.show()

#housing_with_id = housing_df.reset_index()
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.3, "index")

#najprostszy sposob na podzial danych
#train_set, test_set = train_test_split(housing_df, test_size=0.3, shuffle=True, random_state=0)

housing_df['income_cat'] = pd.cut(housing_df['median_income'], 
                                            bins=[0., 1.5, 3., 4.5, 6., np.inf],
                                            labels=[1, 2, 3, 4, 5])

"""DZIELENIE DANYCH NA ZBIORY PRZEZ LOSOWANIE WARSTWOWE"""
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_idx, test_idx in split.split(housing_df, housing_df['income_cat']):
    train_set = housing_df.loc[train_idx]
    test_set = housing_df.loc[test_idx]

for set_ in (train_set, test_set):
    set_.drop('income_cat', axis=1, inplace=True)
housing_copy = train_set

# housing_copy.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5,
#                   s=housing_copy['population']/100, label='populacja', figsize=(10, 7),
#                   c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()

housing_labels = housing_copy['median_house_value'].copy()
housing_copy = housing_copy.drop(columns=['median_house_value'])
housing_num = housing_copy.drop(columns=['ocean_proximity'])

"""RĘCZNE TRANSFORMACJE, POTEM ROBI TO PIPELINE"""
# imputer = SimpleImputer(strategy='median')
# imputer.fit(housing_num)
# X = imputer.transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# housing_cat = housing_copy[['ocean_proximity']]
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)


"""WIELKA TRANSFORMACJA"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaller', StandardScaler()),
])

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing_copy)


""" 
def display_scores(scores):
    print("wyniki: ", scores)
    print("srednia: ", scores.mean())
    print("odchylenie standardowe: ", scores.std())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)

import joblib
joblib.dump(forest_reg, 'random_forest_model.pkl')
print('model zapisano pomyslnie')
#joblib.load('random_forest_model.pkl')
"""

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

print("rozpoczynam nauke\n")
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

import joblib
joblib.dump(forest_reg, 'random_forest_model.pkl')
print('model zapisano pomyslnie')
joblib.dump(grid_search, 'grid_search_cv.pkl')
print('grid_search zapisano pomyslnie')
#joblib.load('random_forest_model.pkl')

final_model = grid_search.best_estimator_
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

from sklearn.metrics import mean_squared_error
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('final_rmse: ', final_rmse)

from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors))))