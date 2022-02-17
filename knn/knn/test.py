from scipy.spatial.distance import cdist
from sklearn.model_selection import LeaveOneOut

from distances import euclidean_distance
from distances import cosine_distance
import numpy as np

from classification import KNNClassifier
from model_selection import knn_cross_val_score
from nearest_neighbors import NearestNeighborsFinder

# # = np.eye(N=3, M=5) / np.sqrt(2)
# # shape = (len(x), len(x),)
#
# # xx_pred = euclidean_distance(x, x)
# # xx_true = np.ones(shape) - np.eye(*shape)
#
#
# x = np.eye(N=3, M=5) / np.sqrt(2)
# shape = (len(x), len(x),)
#
# # xx_pred = cosine_distance(x, x)
# xx_true = np.ones(shape) - np.eye(*shape)
# d = x @ x.T
# norm_x = np.sqrt((x * x).sum(0, keepdims=True))
# x_2 = np.expand_dims(x, axis=1)
#
# print(cosine_distance(x, x))
# #print(xx_true)

# seed = np.random.RandomState(9872)
# X = seed.permutation(500).reshape(10, -1)
# X_train, X_test = X[:4], X[6:]
#
# nn = NearestNeighborsFinder(n_neighbors=3, metric='euclidean')
# nn.fit(X_train)
#
# distances_pred, indices_pred = nn.kneighbors(X_test, return_distance=True)
#
# print(distances_pred, cdist(X_test, X_train))


#
# seed = np.random.RandomState(2789)
# X = seed.permutation(500).reshape(10, -1)
# X_train, X_test = X[:4], X[6:]
# y_train = [0, 0, 1, 1]
#
# clf = KNNClassifier(n_neighbors=3, algorithm='my_own', metric='euclidean', weights='uniform')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# y_true = np.asarray([1, 0, 0, 1])
# from tests.test_model_selection import complex_roots, knn_cross_val_score_sklearn
#
# seed = np.random.RandomState(228)
# x = complex_roots(6)
# x = np.vstack([np.real(x), np.imag(x)]).T
# x += seed.random(x.shape) * 0.2
#
# y = np.ones(len(x), dtype=int)
# y[:len(y) // 2] = 0
#
# cv = LeaveOneOut()
#
# scores_pred = knn_cross_val_score(x, y, k_list=range(1, len(x), 2), cv=cv, scoring='accuracy')
# scores_true = knn_cross_val_score_sklearn(
#     x, y, k_list=range(1, len(x), 2), cv=cv, scoring='accuracy',
#     metric='euclidean', weights='uniform', algorithm='brute',
# )
#
# scores_pred = {k: list(v) for k, v in scores_pred.items()}
# scores_true = {k: list(v) for k, v in scores_true.items()}

x = [[2, 5, 1, 1, 1, 1, 4, 0],
     [3, 5, 6, 3, 1, 1, 0, 0],
     [4, 5, 4, 7, 2, 3, 1, 1]]
x = np.asarray(x)

xx_pred = euclidean_distance(x, np.zeros_like(x)[:1])
xx_true = np.asarray([7, 9, 11])[:, None]
import ipdb; ipdb.set_trace()