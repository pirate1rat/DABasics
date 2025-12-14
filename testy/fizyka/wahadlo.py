import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


X = [0.516, 0.450, 0.410, 0.480, 0.495, 0.284]
y = [2.074, 1.724, 1.608, 1.932, 1.943, 1.119]

x_err = [0.004, 0.003, 0.003, 0.003, 0.003, 0.0005]
y_err = [0.084, 0.033, 0.0061, 0.0035, 0.0013, 0.0051]

from sklearn.linear_model import LinearRegression
X_train = np.array(X).reshape(-1, 1)
y_train = np.array(y).reshape(-1, 1)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_, model.intercept_)

from sklearn.metrics import mean_squared_error
preds = model.predict(X_train)
print(np.sqrt(mean_squared_error(y, preds)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y)
plt.errorbar(X, y, xerr=x_err, yerr=y_err, fmt='none', ecolor='gray', capsize=4)
plt.xlabel(r"Długość wahadła $l$ [m]")
plt.ylabel(r"Kwadrat Okreu $T²$ [s²]")
plt.title(r"Zależność $T²(l)$ z niepewnościami pomiarowymi")

x = np.linspace(min(X), max(X), 100)
prosta = model.coef_ * x + model.intercept_
prosta = prosta.reshape(-1, 1)
plt.plot(x, prosta, color='gray', linestyle='--', label='Dopasowana prosta')
plt.grid(True)
plt.show()