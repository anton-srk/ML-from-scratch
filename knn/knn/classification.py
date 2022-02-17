import numpy as np

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights
        self._labels = None
        self.eps = 1e-5

    def fit(self, train, y=None):
        self._finder.fit(train)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        y_pred = self._labels[indices]
        return np.argmax(self._bincount2d(y_pred, distances), axis=1)

    def _bincount2d(self, arr, distances):
        N = arr.max() + 1
        id = arr + (N * np.arange(arr.shape[0]))[:, None]
        if self._weights != 'uniform':
            bc = np.bincount(id.ravel(), minlength=N*arr.shape[0],
                             weights=1/(distances.ravel()+self.eps)).reshape(-1, N)
        else:
            bc = np.bincount(id.ravel(), minlength=N * arr.shape[0]).reshape(-1, N)
        return bc

    def kneighbors(self, test, return_distance=False):
        return self._finder.kneighbors(test, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedMixin:
    def __init__(self):
        self.batch_size = None

    def kneighbors(self, X, return_distance=False):
        if not hasattr(self,  'batch_size'):
            self.batch_size = None

        batch_size = self.batch_size or X.shape[0]

        distances, indices = [], []

        for i_min in tqdm(list(range(0, X.shape[0], batch_size))):
            i_max = min(i_min + batch_size, X.shape[0])
            X_batch = X[i_min:i_max]

            indices_ = super().kneighbors(X_batch, return_distance=return_distance)
            if return_distance:
                distances_, indices_ = indices_
            else:
                distances_ = None

            indices.append(indices_)
            if distances_ is not None:
                distances.append(distances_)

        indices = np.vstack(indices)
        distances = np.vstack(distances) if distances else None

        result = (indices,)
        if return_distance:
            result = (distances,) + result
        return result if len(result) > 1 else result[0]

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


class BatchedKNNClassifier(BatchedMixin, KNNClassifier):
    def __init__(self, *args, **kwargs):
        KNNClassifier.__init__(self, *args, **kwargs)
        BatchedMixin.__init__(self)
