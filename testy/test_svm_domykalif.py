import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data = np.c_[X_train, y_train]

# Attribute Information:
#     - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude


df = pd.DataFrame(data, columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Label'])
#print(df.info())
# print(df.head())

# df.hist(bins=50)

# df.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.5,
#                   s=df['Population']/100, label='Population', figsize=(10, 7),
#                   c='Label', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()

# corr_mtrx = df.corr()
# print(corr_mtrx['Label'].sort_values(ascending=False))

# from pandas.plotting import scatter_matrix
# attrbs = ['Label', 'MedInc', 'AveRooms']
# scatter_matrix(df[attrbs], figsize=(10, 8))
# plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import LinearSVR

lin_svr = LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)
LinearSVR(random_state=42)

from sklearn.metrics import mean_squared_error

y_pred = lin_svr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(np.sqrt(mse))


def linear_reg():
    ##############################
    print("test: LinearRegression")
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    #final_predictions = lin_reg.predict(X_test)
    final_mse = mean_squared_error(y_test, lin_reg.predict(X_test))
    final_rmse = np.sqrt(final_mse)
    print('final_mse: ', final_mse)
    print('final_rmse: ', final_rmse)

linear_reg()
    