import numpy as np
import matplotlib.pyplot as plt

if 1:
    X_train = np.array([1.943333333, 2.883333333, 3.863333333, 5.8, 6.73, 9.663333333, 11.51333333]).reshape(7, 1)
    y_train = np.array([427.5333333, 441.15, 457.805, 464.0, 471.1, 487.9983333, 498.5273333]).reshape(7, 1)
else:
    X_train = np.array([1.943333333, 3.863333333, 5.8, 6.73, 9.663333333, 11.51333333]).reshape(6, 1)
    y_train = np.array([427.5333333, 457.805, 464.0, 471.1, 487.9983333, 498.5273333]).reshape(6, 1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X_train)


def show_charts(X, y, model):
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(X, y)
    #axis.set(title='dane wielomianowe')
    axis.set_xlabel(xlabel=r"$d_{śr}$ [mm]", fontsize=18)
    axis.set_ylabel(ylabel=r"$Cd$ [mm·pF]", fontsize=18)

    X_new = np.linspace(0, 12, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new_preds = model.predict(X_new_poly)
    axis.plot(X_new, y_new_preds, 'r-', linewidth='2', label='Predicions')

    print(model.predict(poly_features.transform([[0]])))
    print(
        model.predict(poly_features.transform([[1.943333333]]))-y_train[0],
        model.predict(poly_features.transform([[3.863333333]]))-y_train[1])

    plt.show()


def plot_learning_curves(model, X, y):
    model.fit(X, y)
    print("przewidziane parametry: ", model.coef_, model.intercept_)

model = LinearRegression()
plot_learning_curves(model, X_poly, y_train)
show_charts(X_train, y_train, model)

from sklearn.metrics import mean_squared_error
preds = model.predict(X_poly)
print("error: ", np.sqrt(mean_squared_error(y_train, preds)))

plt.show()