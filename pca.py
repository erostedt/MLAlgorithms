import numpy as np


def apply_pca(X, n_components, scale=True):
    means = np.mean(X, axis=1, keepdims=True)
    X_center = X - means

    if scale:
        stds = np.std(X_center, axis=1, keepdims=True)
        X_center = X_center / stds

    *_, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:n_components]

    return X_center @ components
     

class PCA:

    def __init__(self, n_components, scale=True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X):
        n_samples, _ = X.shape
        self._means = np.mean(X, axis=1, keepdims=True)
        self._stds = None
        X_center = X - self._means
        if self.scale:
            self._stds = np.std(X_center, axis=1, keepdims=True)
            X_center = X_center / self._stds
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components = Vt[:self.n_components]
        self.explained_variance = (S ** 2) / (n_samples - 1)
        self.total_variance = np.sum(self.explained_variance)

    def transform(self, X):
        X_center = X - self._means
        if self.scale:
            X_center = X_center / self._stds

        return X_center @ self.components
    
