import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(
        ranks: np.ndarray, top: int, return_ind=True
        ):
    """
    Returns indices of top largest values in rows of array.

    Parameters
    ----------
    ranks : np.ndarray, required
        Input array.
    top : int, required
        Number of largest values.
    return_ind: whether to return indices
    """
    if ranks.shape[1] <= top:
        top_ = ranks.shape[1] - 1
        sorted_r = np.argpartition(ranks, top_, axis=1)
    else:
        sorted_r = np.argpartition(ranks, top, axis=1)[:, :top]

    fdim = np.arange(ranks.shape[0])[:, None]
    so = np.argsort(ranks[fdim, sorted_r], axis=1)

    if return_ind:
        return np.take_along_axis(ranks, sorted_r[fdim, so], axis=1), sorted_r[fdim, so]
    else:
        return sorted_r[fdim, so]


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self._train = None

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, train, y=None):
        self._train = train
        return self

    def kneighbors(self, test, return_distance=False):
        distances = self._metric_func(test, self._train)

        distances_true, indices_true = get_best_ranks(distances, self.n_neighbors)

        if return_distance:
            return distances_true, indices_true
        else:
            return indices_true
