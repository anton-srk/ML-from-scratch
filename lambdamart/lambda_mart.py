import pickle
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


def dcg(ys_true, ys_pred):
    _, argsort = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[argsort]
    ret = 0
    for i, l in enumerate(ys_true_sorted, 1):
        ret += (2 ** l - 1) / np.log2(1 + i)
    return ret


class Solution:
    def __init__(
            self, n_estimators: int = 100, lr: float = 0.25, ndcg_top_k: int = 10,
            subsample: float = 0.6, colsample_bytree: float = 0.9,
            max_depth: int = 6, min_samples_leaf: int = 8
    ):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.best_ndcg = 0
        self.best_tree = 0
        self.trees = []

    @staticmethod
    def _get_data() -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train, X_test = (
            self._scale_features_in_query_groups(X_train, self.query_ids_train),
            self._scale_features_in_query_groups(X_test, self.query_ids_test)
        )
        self.X_train, self.ys_train = torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(-1)
        self.X_test, self.ys_test = torch.Tensor(X_test), torch.Tensor(y_test).unsqueeze(-1)
        
        return None

    @staticmethod
    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        sc = StandardScaler()
        for query in np.unique(inp_query_ids):
            inp_feat_array[inp_query_ids==query] = \
                sc.fit_transform(inp_feat_array[inp_query_ids==query])
        
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        np.random.seed(cur_tree_idx)
        
        lambdas = torch.zeros_like(self.ys_train)
        
        for group in np.unique(self.query_ids_train):
            idx = self.query_ids_train == group
            lambdas[idx, :] = self._compute_lambdas(self.ys_train[idx], train_preds[idx])

        one_t = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
        )
        
        num_cols = int(self.X_train.shape[1] * self.colsample_bytree)
        num_rows = int(self.X_train.shape[0] * self.subsample)
        
        sub_idx = np.random.choice(
            np.arange(self.X_train.shape[0]), num_rows,
            replace=False
        )
        col_idx = np.random.choice(
            np.arange(self.X_train.shape[1]), num_cols,
            replace=False
        )
        X_sub, y_sub = self.X_train[sub_idx][:, col_idx], self.ys_train[sub_idx, :]
        
        lambdas[torch.isnan(lambdas)] = 0
        one_t.fit(X_sub, lambdas[sub_idx])

        return one_t, col_idx

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcgs = []
        for group in np.unique(queries_list):
            idx = queries_list == group
            ndcgs += [self._ndcg_k(true_labels[idx], preds[idx], ndcg_top_k=self.ndcg_top_k)]

        return np.mean(ndcgs)

    def fit(self):
        np.random.seed(0)
        self.trees = []
        self.col_idxs = []
        for i in tqdm(list(range(self.n_estimators))):
            if i == 0:
                train_preds = torch.zeros_like(self.ys_train)
                test_preds = torch.zeros_like(self.ys_test)
            else:
                train_preds -= \
                    self.lr * torch.Tensor(
                        self.trees[-1].predict(self.X_train[:, self.col_idxs[-1]])
                    ).unsqueeze(-1)
                test_preds -= \
                    self.lr * torch.Tensor(
                        self.trees[-1].predict(self.X_test[:, self.col_idxs[-1]])
                    ).unsqueeze(-1)
                
            t_ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds)
            if t_ndcg > self.best_ndcg:
                self.best_ndcg = t_ndcg
                self.best_tree = i

            print(self.best_ndcg)
            
            tr, col_idx = self._train_one_tree(i, train_preds)

            self.trees += [tr]
            self.col_idxs += [col_idx]
            
        return self.trees, self.col_idxs

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros(size=(data.shape[0], 1))
        for i, tree in enumerate(self.trees[:self.best_tree]):
            preds -= self.lr * torch.Tensor(tree.predict(data[:, self.col_idxs[i]])).unsqueeze(-1)
        return preds

    def _ndcg_k(
        self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
        ndcg_top_k: int
    ) -> float:
        
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        pred_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)

        return (pred_dcg / ideal_dcg).numpy()[0][0]

    def _dcg_k(self, ys_true, ys_pred, k):
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort][:k, :]
        ret = 0
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    def save_model(self, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(
                (self.trees, self.col_idxs, self.lr, self.best_tree),
                handle
            )

    def load_model(self, path: str):
        with open(path, 'rb') as handle:
            self.trees, self.col_idxs, self.lr, self.best_tree = pickle.load(handle)
    
    def _compute_lambdas(
        self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor
    ) -> torch.FloatTensor:

        ideal_dcg = dcg(y_true, y_true)
        N = 1 / ideal_dcg

        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            Sij = self._compute_labels_in_batch(y_true)
            gain_diff = self._compute_gain_diff(y_true)

            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))

            delta_ndcg = torch.abs(N * gain_diff * decay_diff)

            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update

    @staticmethod
    def _compute_labels_in_batch(y_true):
        rel_diff = y_true - y_true.t()
        pos_pairs = (rel_diff > 0).type(torch.float32)
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        
        return Sij

    @staticmethod
    def _compute_gain_diff(y_true):
        return torch.pow(2, y_true) - torch.pow(2, y_true.t())
