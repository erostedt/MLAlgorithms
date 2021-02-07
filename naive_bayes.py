import numpy as np

def naive_bayes(X_train, y_train, X_test):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)

    means = np.zeros(shape=(n_classes, n_features))
    vars = np.zeros_like(means)
    log_priors = np.zeros((n_classes, ))

    # Fit model by finding means, variances and priors for each class gaussian.
    for c in classes:
        X_class = X_train[y_train == c]
        means[c, :] = np.mean(X_class, axis=0)
        vars[c, :] = np.var(X_class, axis=0)
        log_priors[c] = X_class.shape[0] / n_samples

    log_priors = np.log(log_priors)

    def _log_of_normal_pdf(x, means, vars):
        return -np.log(2 * np.pi * vars) - ((x - means) ** 2) / (2 * vars)

    # Predict using argmax of loglikelihood.    
    return np.array([classes[np.argmax(log_priors + _log_of_normal_pdf(x, means, vars))] for x in X_test])        

    
 

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._means = np.zeros(shape=(n_classes, n_features))
        self._vars = np.zeros_like(self._means)
        self._log_priors = np.zeros((n_classes, ))

        for c in self._classes:
            X_class = X[y == c]
            self._means[c, :] = np.mean(X_class, axis=0)
            self._vars[c, :] = np.var(X_class, axis=0)
            self._log_priors[c] = X_class.shape[0] / n_samples

        self._log_priors = np.log(self._log_priors)

    def predict(self, X):
        """
        Return the argmax of the likelihood estimation for each data point.
        """
        return np.array([self._classes[np.argmax(self._log_priors + self._log_of_normal_pdf(x))] for x in X])        

    def _log_of_normal_pdf(self, x):
        return -np.log(2 * np.pi * self._vars) - ((x - self._means) ** 2) / (2 * self._vars)
