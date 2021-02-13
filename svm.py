import numpy as np
import matplotlib.pyplot as plt

class SVM:

    def __init__(self, learning_rate=1e-3, max_iter=1000, reguliser=1, kernel=None):
        if kernel is None:
            self.kernel = lambda x, y: np.sum(x * y, axis=1)
        
        self.learning_rate = learning_rate
        self.reguliser = reguliser
        self.weights = None
        self.bias = None
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))

        # Put bias inside weights
        X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)
        self.weights = np.zeros(shape=(n_classes+1, 1))

        for _ in self.max_iter:
            self.weights -= self._hinge_gradient(X, y)
        
        self.X = X
        self.y = y

    def predict(self, X):
        n_samples, _ = X.shape
        X = np.concatenate((X, np.ones(shape=(n_samples, 1))), axis=1)
        return np.sign(self.kernel(X, self.weights))
    
    def _hinge_gradient(self, X, y):
        N = len(y)
        signs = np.maximum(1 - y * self.kernel(X, self.weights), np.zeros_like(y))
        grad = self.reguliser * np.ones(shape=(N, len(self.weights))) * self.weights[np.newaxis, :]
        grad[signs == 0] -= y * X
        return np.sum(grad, axis=0) / N

    def plot_fit(self):
        if len(self.weights) != 3:
            raise NotImplementedError('Only implemented for 2d')
        
        x_window = np.min(self.X[:, 0]), np.max(self.X[:, 0])

        plt.scatter(self.X[:, 0], self.X[:, 1])

        
        k, m = self._abc_to_km(self.weights)
        xs = np.linspace(*x_window, 100)
        plt.plot(xs, k * xs + m)
        plt.show()
        
    @staticmethod
    def _abc_to_km(line_coeffs):
        # ax + by + c = 0 -> y = kx + m. i.e k = -a/b and m = -c/b
        a, b, c = line_coeffs
        k = -a / b
        m = -c / b
        return k, m