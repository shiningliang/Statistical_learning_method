import sys
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as matplot
import numpy as np
from scipy.optimize import fmin_bfgs
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# class IrisDatasetLogisticRegression:
#     """
#      Implementing logistic regression using Iris dataset
#     """
#
#     """Global class variables"""
#     "Matrix containing set of features"
#     X = None
#
#     "Matrix containing set of outputs"
#     y = None
#
#     def __init__(self, X, y):
#         """ USAGE:
#         Default constructor
#
#         PARAMETERS:
#         X - feature matrix
#         y - output matrix
#
#         RETURN:
#         """
#         self.X = X
#         self.y = y
#
#         """Convert y into a proper 2 dimensional array/matrix. This is to facilitate proper matrix arithmetics."""
#         if len(y.shape) == 1:
#             y.shape = (y.shape[0], 1)
#
#     def sigmoid(self, z):
#         """ USAGE:
#         Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
#
#         PARAMETERS:
#         z - Matrix, vector or scalar
#
#         RETURN:
#         The sigmoid value
#         """
#         return 1.0 / (1.0 + np.exp(-z))
#
#     def compute_cost(self, X, y, theta):
#         """ USAGE:
#         Define the loss function
#
#         PARAMETERS:
#         X - Features
#         y - Output
#         theta
#
#         RETURN:
#         return the loss function value
#         """
#         m = X.shape[0]
#         z = np.dot(X, theta)
#         h = self.sigmoid(z)
#
#         J = (float(-1) / m) * ((y.T.dot(np.log(h))) + ((1 - y.T).dot(np.log(1 - h))))
#         return J
#
#     def compute_gradient(self, X, y, theta):
#         """ USAGE:
#         Compute the gradient using vectorization.
#
#         PARAMETERS:
#         X - Features
#         y - Output
#         theta
#
#         RETURN:
#         """
#         m = X.shape[0]
#         z = np.dot(X, theta)
#         h = self.sigmoid(z)
#
#         grad = (float(1) / m) * ((h - y).T.dot(X))
#         return grad
#
#     def plot_two_features(self):
#         """ USAGE:
#         Plot first two features from the Iris dataset
#
#         PARAMETERS:
#
#         RETURN:
#         """
#         fig = plt.figure()
#         ax = fig.add_subplot(111, title="Iris Dataset - Plotting two features", xlabel='Sepal Length',
#                              ylabel='Sepal Width')
#         plt.setp(ax.get_xticklabels(), visible=False)
#         plt.setp(ax.get_yticklabels(), visible=False)
#
#         setosa = np.where(self.y == 0)
#         versicolour = np.where(self.y == 1)
#
#         ax.scatter(X[setosa, 0], X[setosa, 1], s=20, c='r', marker='o')
#         ax.scatter(X[versicolour, 0], X[versicolour, 1], s=20, c='r', marker='x')
#         plt.legend(('Iris Type - Setosa', 'Iris Type - Versicolour'))
#         plt.show()
#
#     def plot_three_features(self):
#         """ USAGE:
#         Plot first two features from the Iris dataset
#
#         PARAMETERS:
#
#         RETURN:
#         """
#         fig = plt.figure()
#         ax = fig.add_subplot(111, title="Iris Dataset - Plotting three features", xlabel='Sepal Length',
#                              ylabel='Sepal Width', zlabel='Petal Length', projection='3d')
#         plt.setp(ax.get_xticklabels(), visible=False)
#         plt.setp(ax.get_yticklabels(), visible=False)
#         plt.setp(ax.get_zticklabels(), visible=False)
#
#         ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, c='r', marker='o')
#         plt.show()
#
#     def run_logistic_regression(self):
#         """ USAGE:
#         Apply principles of logistic regression
#
#         PARAMETERS:
#
#         RETURN:
#         """
#
#         """m= number of training data, n= number of features"""
#         m = X.shape[0]
#         n = X.shape[1]
#
#         """Add intercept term (b) to X"""
#         self.X = np.hstack((np.ones((m, 1)), self.X))
#
#         """Initialize fitting parameters. Take into account the intercept term."""
#         initial_theta = np.zeros((n + 1, 1))
#
#         """"Compute initial loss and gradient"""
#         loss = self.compute_cost(self.X, self.y, initial_theta)
#         gradient = self.compute_gradient(self.X, self.y, initial_theta)
#
#         print('Cost at initial theta (zeros): {0} \nGradient at initial theta (zeros): {1}'.format(loss, gradient))
#
#         def f(theta):
#             return np.ndarray.flatten(self.compute_cost(self.X, self.y, initial_theta))
#
#         def fprime(theta):
#             return np.ndarray.flatten(self.compute_gradient(self.X, self.y, initial_theta))
#
#         print(fmin_bfgs(f, initial_theta, fprime, disp=True, maxiter=400, full_output=True, retall=True))

class LogisticRegressionBFGS:
    """ An implementation of logistic regression. """

    def __init__(self, x, y, lambda_=0.0):
        self.x = x
        self.y = np.atleast_2d(y).transpose()
        self._lambda = lambda_

    def _sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        idx = res == 1
        res[idx] = .99
        return res

    def _compute_cost(self, theta):
        """ Calculate the cost function:
        J = -1 / m * (y' * log(sigmoid(X * theta)) + (1 .- y') * log(1 .- sigmoid(X * theta)))
        J += lambda / (2 * m) * theta(2 : end)' * theta(2 : end)
        """
        m = self.x.shape[0]
        x_bias = np.hstack((np.ones((m, 1)), self.x))
        theta = np.atleast_2d(theta).transpose()

        J = -1.0 / m * (np.dot(self.y.transpose(), np.log(self._sigmoid(np.dot(x_bias, theta))))
                        + np.dot(1 - self.y.transpose(), np.log(1 - self._sigmoid(np.dot(x_bias, theta)))))
        J += self._lambda / (2 * m) * sum(theta[1::] ** 2)

        return J[0, 0]

    def _compute_grad(self, theta):
        """ Calculate the gradient of J:
        grad = 1 / m * (X' * (sigmoid(X * theta) - y))
        grad(2 : end) += lambda / m * theta(2 : end)
        """
        m = self.x.shape[0]
        x_bias = np.hstack((np.ones((m, 1)), self.x))
        theta = np.atleast_2d(theta).transpose()

        grad = 1.0 / m * (np.dot(x_bias.transpose(), self._sigmoid(np.dot(x_bias, theta)) - self.y))
        grad[1::] += self._lambda / m * theta[1::]

        return grad.ravel()

    def learn(self, max_iter=300):
        """ Train theta from the dataset, return value is a 1-D array.
        """
        initial_theta = [0] * (self.x.shape[1] + 1)
        args_ = ()
        theta = fmin_bfgs(f=self._compute_cost, x0=initial_theta,
                          fprime=self._compute_grad, args=args_, maxiter=max_iter)
        self._theta = np.atleast_2d(theta).transpose()

    def predict(self, x):
        m = x.shape[0]
        x_bias = np.hstack((np.ones((m, 1)), x))
        p = np.zeros((m, 1))
        prob = self._sigmoid(np.dot(x_bias, self._theta))
        idx = prob >= 0.5
        p[idx] = 1

        return p.ravel()


if __name__ == '__main__':
    try:
        iris = datasets.load_iris()
        X = iris.data
        Y = iris.target
        train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

        sc = StandardScaler()
        sc.fit(train_X)
        train_X_std = sc.transform(train_X)
        test_X_std = sc.transform(test_X)

        clf = LogisticRegressionBFGS(train_X_std, train_y, 0.1)
        clf.learn()
        pred_y = clf.predict(test_X_std)
        print('LR BFGS Acc is : ', accuracy_score(test_y, pred_y))

        clf = LogisticRegression()
        clf.fit(train_X_std, train_y)
        pred_y = clf.predict(test_X_std)
        print('LR SGD Acc is : ', accuracy_score(test_y, pred_y))

    except:
        print("unexpected error:", sys.exc_info())
