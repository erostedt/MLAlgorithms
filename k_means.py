import numpy as np
import matplotlib.pyplot as plt
from MLAlgorithms.utils import random_colour


class KMeans:
	"""
	Assumes Labeling 0 -> k-1 
	"""

	def __init__(self, k=2, max_iter=100):
		self.k = k
		self.max_iter = max_iter
		self.centers = None
		self.clusters = None

	def fit(self, X):
		n_samples, _ = X.shape
		self.centers = np.random.choice(X, self.k, replace=False)
		self.clusters = [0] * self.k 
		distances = np.zeros(shape=(n_samples, self.k))

		for _ in self.max_iter:

			for col, center in enumerate(self.centers):
				distances[:, col] = np.sum(np.square(X - center), axis=1)

			cluster_idxs = np.argmin(distances, axis=1)

			for k in range(self.k):
				self.clusters[k] = X[cluster_idxs == k]
				self.centers[k] = np.mean(self.clusters[k], axis=0)

	def predict(self, X):
		n_samples, _ = X.shape
		distances = np.zeros(shape=(n_samples, self.k))

		for col, center in enumerate(self.centers):
			distances[:, col] = np.sum(np.square(X - center), axis=1)

		return np.argmin(distances, axis=1)

	def plot_clusters(self):
		_, ax = plt.subplot()
		for idx, center, cluster in enumerate(zip(self.centers, self.clusters)):
			colour = random_colour()
			ax.scatter(center[0], center[1], c=colour, s=50, marker='*')
			ax.scatter(cluster[:, 0], cluster[:, 1], c=colour, label=f'Cluster: {idx}')

		ax.set_title(f'{self.k}-NN clustering.')
		ax.set_xlabel('Feature 1')
		ax.set_ylabel('Feature 2')
		ax.legend()
		plt.show()















