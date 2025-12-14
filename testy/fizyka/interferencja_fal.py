import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator



lamb = [171.5, 156.0, 143.6, 132.333, 123.0, 114.333, 108.0, 100.333, 95.333, 90.667]
X = [2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0, 3800.0]
y = [lamb[i]*X[i]/1000 for i in range(10)]

x_err = [0.003] * len(X)
y_err = [0.012] * len(y)

# from sklearn.linear_model import LinearRegression
# X_train = np.array(X).reshape(-1, 1)
# y_train = np.array(y).reshape(-1, 1)
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(model.coef_, model.intercept_)

# from sklearn.metrics import mean_squared_error
# preds = model.predict(X_train)
# print(np.sqrt(mean_squared_error(y, preds)))


fig, ax = plt.subplots()
ax.scatter(X, y)
plt.axhline(y=343.677, color='gray', linestyle='--', linewidth=1)
plt.xlabel(r"Częstotliwość $f$ [Hz]")
plt.ylabel(r"Prędkość dźwięku [m/s²]")
plt.title(r"Wykres zależność prędkości dźwięku od częstotliwości")
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.set_ylim(330, 360)
#ax.yaxis.set_major_locator(MultipleLocator(10))
plt.grid(True)

# x = np.linspace(min(X), max(X), 100)
# prosta = model.coef_ * x + model.intercept_
# prosta = prosta.reshape(-1, 1)
# plt.plot(x, prosta, color='gray', linestyle='--', label='Dopasowana prosta')
plt.show()