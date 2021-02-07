import numpy as np
from collections import Counter


class KNN:

	def __init__(self, k=1):
		self.k = k
		self.X = None
		self.y = None

	def fit(self, X, y):
		self.X = X
		self.y = y

	def predict(self, X):
		n_samples, _ = X.shape
		predictions = np.zeros(shape=(n_samples, ))
		for i, x in enumerate(X):
			distances = np.sum(np.square(self.X - x), axis=1)
			closest_idxs = np.argsort(distances)[:self.k]
			labels = self.y[closest_idxs]
			predictions[i] = self._majority_vote(labels)

		return predictions

	@staticmethod
	def _majority_vote(arr):
		return Counter(arr).most_common(1)[0][0]


def knn_one_liner(X_train, y_train, X_test, k):
	return np.array([Counter(y_train[np.argsort(np.sum(np.square(X_train - x), axis=1))[:k]]).most_common(1)[0][0] for x in X_test])
