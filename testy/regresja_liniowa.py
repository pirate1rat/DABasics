import numpy as np
import matplotlib.pyplot as plt

samples = 100
bonus = 100

np.random.seed(42)
X = 10 * np.random.rand(samples, 1)
y = 0.5 * X + 5 + np.random.randn(samples, 1)

for i in range(3):
    y[np.random.randint(0.1 * samples, 0.8 * samples)] += bonus

fig, axes = plt.subplots(figsize=(10, 6))
axes.scatter(X, y)
axes.set(title='dane liniowe')
axes.set_xlabel(xlabel='X', fontsize=18)
axes.set_ylabel(ylabel='y', fontsize=18, rotation=0)
#plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.linear_model import LinearRegression, SGDRegressor

model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_, model.intercept_)
print(model.predict([[0], [10]]))
axes.plot([0, 10], model.predict([[0], [10]]), color='red')
plt.show()

# plt.plot(X, y, 'b.')
# plt.xlabel("$x$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 10, 0, 15])
# plt.show()