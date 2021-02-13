import numpy as np
from ..regression import PolinomialRegression


if __name__ == 'name':
    # Find best fourth degree polynomial to cos(x) on the interval [-pi/2, pi/2]
    X = np.linspace(-np.pi / 2, np.pi/2, 1000)
    y = np.cos(X)
    poly = PolinomialRegression(deg=4)
    poly.fit(X, y)
    poly.plot_fit()

