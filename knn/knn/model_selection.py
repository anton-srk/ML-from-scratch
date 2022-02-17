from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import KNNClassifier, BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, b_size=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    acc_dict = defaultdict(list)
    for train_idx, test_idx in cv.split(X, y):
        knn = BatchedKNNClassifier(n_neighbors=np.max(k_list), **kwargs)
        knn.set_batch_size(b_size)
        knn.fit(X[train_idx], y[train_idx])
        dist, ind = knn.kneighbors(X[test_idx], return_distance=True)
        for k_val in k_list:
            acc_dict[k_val] += [scorer(y[test_idx], knn._predict_precomputed(ind[:, :k_val], dist[:, :k_val]))]

    return acc_dict
