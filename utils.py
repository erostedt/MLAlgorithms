import numpy as np
import os


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

def points_to_vectors(points):
	return tuple(zip(*points))

def random_colour():
    return np.random.choice(np.linspace(0, 1, 256), size=(3, ), replace=True)
