import numpy as np
import matplotlib.pyplot as plt
from MLAlgorithms.utils import points_to_vectors


class PolinomialRegression:

	def __init__(self, deg=1, method='gd', max_iter=1000, learning_rate=1e-2):
		"""

		methods: gradient_descent: gd, pseudo_inverse: pi
		"""
		methods = ('gd', 'pi')
		if method not in methods:
			raise NotImplementedError('Only Implemented for methods: gd and pi')
		
		self.deg = deg
		self.method = method
		self.weights = np.random.random(deg+1)
		self.max_iter = max_iter
		self.learning_rate = learning_rate
		self.X = None
		self.y = None

	def __repr__(self):
		polynomial_strings = [str(self.weights[0])] + [f'{w:.2f} x**{i}' for i, w in enumerate(self.weights[1:], start=1)]
		return ' + '.join(polynomial_strings)

	def fit(self, X, y):
		self.X, self.y = X, y

		V = self._vandermonde(X, self.deg)

		if self.method == 'pi':
			self.weights = np.linalg.solve(V.T @ V, V.T @ y)

		else:
			gradient = lambda w: V @ w - y
			for _ in range(self.max_iter):
				grad = gradient(self.weights)
				if grad < 1e-6:
					break
				
				else:
					self.weights -= self.learning_rate * grad
				
				
	def predict(self, X):
		V = self._vandermonde(X, self.deg)
		return V @ self.weights

	def plot_fit(self):
		X_sorted, y_sorted = points_to_vectors(sorted(zip(self.X, self.y), key=lambda item: item[0]))
		y_predictions = self.predict(X_sorted)

		plt.plot(X_sorted, y_sorted, c='b')
		plt.plot(X_sorted, y_predictions, 'r--')
		plt.title(f'Polynomial fitted line: {self.repr()}')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend(['Data', 'Fit'])
		plt.show()

	@staticmethod
	def _vandermonde(xs, deg):
		N = len(xs)
		vandermonde = np.ones((N, deg))
		for order in range(1, deg + 1):
			vandermonde[:, order] = xs ** order
		return vandermonde


class LogisticRegression:

	def __init__(self, max_iter=1000, learning_rate=1e-2):
		self.max_iter = max_iter
		self.learning_rate = learning_rate
		self.weights = None
		self.bias = None
		self.X = None
		self.y = None

	def __repr__(self):
		return f'Weights: {self.weights} \n Bias: {self.bias}'

	def fit(self, X, y):
		self.X, self.y = X, y

		n_samples, n_features = X.shape

		self.weights = np.random.random(n_features)
		self.bias = np.random.random()

		for _ in range(self.max_iter):
			predictions = self._sigmoid(X @ self.weights + self.bias)
			errors = predictions - y

			dw = X.T @ (errors) / n_samples
			db = np.sum(errors) / n_samples

			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db


	def predict(self, X):
		predictions = self._sigmoid(X @ self.weights + self.bias)
		class_predictions = np.where(predictions >= 0.5, 1, 0)
		return class_predictions

	def plot_fit(self):
		X_sorted, y_sorted = points_to_vectors(sorted(zip(self.X, self.y), key=lambda item: item[0]))
		y_predictions = self.predict(X_sorted)

		plt.plot(X_sorted, y_sorted, c='b')
		plt.plot(X_sorted, y_predictions, 'r--')
		plt.plot(X_sorted, X_sorted @ self.weights + self.bias)
		plt.title(f'Logistic regression: {self.repr()}')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend(['Data', 'Fit', 'Sigmoind'])
		plt.show()


	@staticmethod
	def _sigmoid(z):
		return 1 / (1 + np.exp(-z))








		