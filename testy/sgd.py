import numpy as np
import matplotlib.pyplot as plt


samples = 50

np.random.seed(42)
X = 10 * np.random.rand(samples, 1)
y = 1 * X + 5 + np.random.randn(samples, 1)

fig, axis = plt.subplots(figsize=(10, 6))
axis.scatter(X, y)
axis.set(title='dane liniowe')
axis.set_xlabel(xlabel='X', fontsize=18)
axis.set_ylabel(ylabel='y', fontsize=18, rotation=0)
#plt.show()


########   WSADOWY PEŁNY
eta = 0.01
iterations = 5000

theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((samples, 1)), X]  # add x0 = 1 to each instance
X_new = np.array([[0], [10]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

# for iter in range(iterations):
#     gradient = 2/samples * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradient
#     print(theta)

# print(theta)

theta_path_bgd = []
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    global iterations
    for iteration in range(iterations):
        if iteration%10==0:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 10, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)
    print(theta)
    plt.show()


############################  STOCHASTYCZNY
theta_path_sgd = []
m = len(X_b)
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:                    # not shown in the book
            y_predict = X_new_b.dot(theta)           # not shown
            style = "b-" if i > 0 else "r--"         # not shown
            plt.plot(X_new, y_predict, style)        # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        #eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)                 # not shown

plt.plot(X, y, "b.")                                 # not shown
plt.xlabel("$x_1$", fontsize=18)                     # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)           # not shown
plt.axis([0, 10, 0, 15])                              # not shown
plt.show()        

#plot_gradient_descent(theta, eta)


# plt.figure(figsize=(10,4))
# plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
# plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
# plt.show()