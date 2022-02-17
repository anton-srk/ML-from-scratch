import numpy as np


def euclidean_distance(x, y):
    P = np.add.outer(np.sum(x ** 2, axis=1), np.sum(y ** 2, axis=1))
    N = np.dot(x, y.T)
    dists = np.sqrt(P - 2 * N)

    return dists


def cosine_distance(x, y):
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    cosine_distance = 1 - np.dot(x, y.T) / np.outer(x_norm, y_norm)

    return cosine_distance
