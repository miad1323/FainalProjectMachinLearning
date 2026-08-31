from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder


@dataclass
class FIGSNode:
    value: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: FIGSNode | None = None
    right: FIGSNode | None = None
    sample_idx: np.ndarray | None = None
    parent_id: int | None = None
    node_id: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class FIGSTree:
    root: FIGSNode
    splits: int = 0


class FIGSRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_splits: int = 8,
        min_samples_leaf: int = 10,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
        max_thresholds: int = 64,
    ):
        self.max_splits = max_splits
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_thresholds = max_thresholds

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        return X

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        vals = np.unique(values[np.isfinite(values)])
        if len(vals) <= 1:
            return np.array([])
        if len(vals) > self.max_thresholds:
            qs = np.linspace(0.02, 0.98, self.max_thresholds)
            vals = np.unique(np.quantile(vals, qs))
        return (vals[:-1] + vals[1:]) / 2.0

    def _mse(self, y: np.ndarray) -> float:
        return float(np.mean((y - np.mean(y)) ** 2)) if len(y) > 0 else 0.0

    def _best_split_for_leaf(self, X, y, idx):
        if len(idx) < 2 * self.min_samples_leaf:
            return None
        parent_mse = self._mse(y[idx])
        best = None
        for j in range(X.shape[1]):
            col = X[idx, j]
            if not np.isfinite(col).any():
                continue
            for thr in self._candidate_thresholds(col):
                left = idx[col <= thr]
                right = idx[col > thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                mse_left = self._mse(y[left])
                mse_right = self._mse(y[right])
                weighted_mse = (len(left) * mse_left + len(right) * mse_right) / len(idx)
                gain = parent_mse - weighted_mse
                if best is None or gain > best[0]:
                    best = (gain, j, float(thr), left, right)
        if best is None or best[0] <= self.min_impurity_decrease:
            return None
        return best

    def _new_tree(self) -> FIGSTree:
        root = FIGSNode(
            value=np.array([0.0]),
            sample_idx=np.arange(self.n_samples_),
            node_id=self._next_node_id()
        )
        return FIGSTree(root=root, splits=0)

    def _next_node_id(self):
        out = self._node_counter
        self._node_counter += 1
        return out

    def _iter_leaves(self, node):
        if node.is_leaf:
            yield node
            return
        yield from self._iter_leaves(node.left)
        yield from self._iter_leaves(node.right)

    def _leaf_indices(self, tree, X):
        node = tree.root
        idx = np.arange(len(X))
        stack = [(node, idx)]
        while stack:
            n, ids = stack.pop()
            if n.is_leaf:
                n.sample_idx = ids
                continue
            mask = X[ids, n.feature] <= n.threshold
            stack.append((n.right, ids[~mask]))
            stack.append((n.left, ids[mask]))

    def _predict_tree(self, tree, X):
        pred = np.zeros(len(X), dtype=float)

        def walk(node, ids):
            if len(ids) == 0:
                return
            if node.is_leaf:
                pred[ids] = node.value[0]
                return
            mask = X[ids, node.feature] <= node.threshold
            walk(node.left, ids[mask])
            walk(node.right, ids[~mask])

        walk(tree.root, np.arange(len(X)))
        return pred

    def predict(self, X):
        X = self._check_X(X)
        if not getattr(self, "tree_", None):
            raise RuntimeError("FIGSRegressor not fitted")
        return self._predict_tree(self.tree_, X)

    def fit(self, X, y, sample_weight=None):
        X = self._check_X(X)
        y = np.asarray(y, dtype=float)
        self.n_samples_ = len(X)
        self.n_features_in_ = X.shape[1]
        self._node_counter = 0
        self.tree_ = self._new_tree()
        self.split_history_ = []
        residual = y.copy()
        pred = np.zeros_like(y)
        for _ in range(self.max_splits):
            residual = y - pred
            candidates = []
            self._leaf_indices(self.tree_, X)
            for leaf in self._iter_leaves(self.tree_.root):
                ids = leaf.sample_idx
                if ids is None or len(ids) < 2 * self.min_samples_leaf:
                    continue
                split = self._best_split_for_leaf(X, residual, ids)
                if split is not None:
                    gain, feat, thr, left_idx, right_idx = split
                    candidates.append((gain, leaf, feat, thr, left_idx, right_idx))
            if not candidates:
                break
            candidates.sort(key=lambda z: z[0], reverse=True)
            gain, leaf, feat, thr, left_idx, right_idx = candidates[0]
            if gain <= self.min_impurity_decrease:
                break
            leaf.feature = feat
            leaf.threshold = thr
            leaf.left = FIGSNode(
                value=np.array([0.0]),
                sample_idx=left_idx,
                parent_id=leaf.node_id,
                node_id=self._next_node_id()
            )
            leaf.right = FIGSNode(
                value=np.array([0.0]),
                sample_idx=right_idx,
                parent_id=leaf.node_id,
                node_id=self._next_node_id()
            )
            leaf.sample_idx = None
            self.tree_.splits += 1
            for child in (leaf.left, leaf.right):
                child.value[0] = np.mean(residual[child.sample_idx])
            pred = self._predict_tree(self.tree_, X)
            self.split_history_.append({
                "gain": float(gain),
                "feature": int(feat),
                "threshold": float(thr)
            })
        self.n_splits_ = len(self.split_history_)
        return self

    @property
    def feature_importances_(self):
        if not self.split_history_:
            return np.zeros(self.n_features_in_)
        imp = np.zeros(self.n_features_in_)
        total = 0.0
        for item in self.split_history_:
            imp[item["feature"]] += max(item["gain"], 0.0)
            total += max(item["gain"], 0.0)
        return imp / total if total else imp

    def export_rules(self, feature_names=None):
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{j}" for j in range(self.n_features_in_)]
        )
        out = []

        def walk(node, path):
            if node.is_leaf:
                cond = " AND ".join(path) if path else "TRUE"
                out.append(f"IF {cond} THEN prediction={node.value[0]:.6f}")
                return
            f = names[node.feature]
            walk(node.left, path + [f"{f} <= {node.threshold:.6g}"])
            walk(node.right, path + [f"{f} > {node.threshold:.6g}"])

        walk(self.tree_.root, [])
        return out


class FIGSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_splits: int = 8,
        min_samples_leaf: int = 8,
        min_impurity_decrease: float = 1e-8,
        random_state: int = 42,
        max_thresholds: int = 64,
    ):
        self.max_splits = max_splits
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_thresholds = max_thresholds

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        return X

    def _candidate_thresholds(self, values: np.ndarray) -> np.ndarray:
        vals = np.unique(values[np.isfinite(values)])
        if len(vals) <= 1:
            return np.array([])
        if len(vals) > self.max_thresholds:
            qs = np.linspace(0.02, 0.98, self.max_thresholds)
            vals = np.unique(np.quantile(vals, qs))
        return (vals[:-1] + vals[1:]) / 2.0

    def _gini_impurity(self, y_idx: np.ndarray) -> float:
        if len(y_idx) == 0:
            return 0.0
        counts = np.bincount(y_idx, minlength=self.n_classes_)
        probs = counts / len(y_idx)
        return 1.0 - np.sum(probs ** 2)

    def _best_split_for_leaf(self, X, y, idx):
        if len(idx) < 2 * self.min_samples_leaf:
            return None
        parent_gini = self._gini_impurity(y[idx])
        best = None
        for j in range(X.shape[1]):
            col = X[idx, j]
            if not np.isfinite(col).any():
                continue
            for thr in self._candidate_thresholds(col):
                left = idx[col <= thr]
                right = idx[col > thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gini_left = self._gini_impurity(y[left])
                gini_right = self._gini_impurity(y[right])
                weighted_gini = (len(left) * gini_left + len(right) * gini_right) / len(idx)
                gain = parent_gini - weighted_gini
                if best is None or gain > best[0]:
                    best = (gain, j, float(thr), left, right)
        if best is None or best[0] <= self.min_impurity_decrease:
            return None
        return best

    def _new_tree(self, class_idx: int) -> FIGSTree:
        root = FIGSNode(
            value=np.array([0.0]),
            sample_idx=np.arange(self.n_samples_),
            node_id=self._next_node_id()
        )
        return FIGSTree(root=root, splits=0)

    def _next_node_id(self):
        out = self._node_counter
        self._node_counter += 1
        return out

    def _iter_leaves(self, node):
        if node.is_leaf:
            yield node
            return
        yield from self._iter_leaves(node.left)
        yield from self._iter_leaves(node.right)

    def _leaf_indices(self, tree, X):
        node = tree.root
        idx = np.arange(len(X))
        stack = [(node, idx)]
        while stack:
            n, ids = stack.pop()
            if n.is_leaf:
                n.sample_idx = ids
                continue
            mask = X[ids, n.feature] <= n.threshold
            stack.append((n.right, ids[~mask]))
            stack.append((n.left, ids[mask]))

    def _predict_tree(self, tree, X):
        pred = np.zeros(len(X), dtype=float)

        def walk(node, ids):
            if len(ids) == 0:
                return
            if node.is_leaf:
                pred[ids] = node.value[0]
                return
            mask = X[ids, node.feature] <= node.threshold
            walk(node.left, ids[mask])
            walk(node.right, ids[~mask])

        walk(tree.root, np.arange(len(X)))
        return pred

    def _predict_raw(self, X):
        return np.column_stack([self._predict_tree(t, X) for t in self.trees_])

    def _softmax(self, raw):
        exp = np.exp(raw - np.max(raw, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        return np.clip(prob, 1e-8, 1 - 1e-8)

    def predict_proba(self, X):
        X = self._check_X(X)
        if not getattr(self, "trees_", None):
            raise RuntimeError("FIGSClassifier not fitted")
        raw = self._predict_raw(X)
        return self._softmax(raw)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]

    def fit(self, X, y, sample_weight=None):
        X = self._check_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = len(X)
        self.n_features_in_ = X.shape[1]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.label_encoder_ = le
        self._node_counter = 0
        self.trees_ = [self._new_tree(k) for k in range(self.n_classes_)]
        self.split_history_ = []
        raw = np.zeros((self.n_samples_, self.n_classes_))
        prob = self._softmax(raw)
        for _ in range(self.max_splits):
            residual = np.zeros_like(prob)
            for k in range(self.n_classes_):
                residual[:, k] = (y_enc == k).astype(float) - prob[:, k]
            candidates = []
            for tree_idx, tree in enumerate(self.trees_):
                self._leaf_indices(tree, X)
                for leaf in self._iter_leaves(tree.root):
                    ids = leaf.sample_idx
                    if ids is None or len(ids) < 2 * self.min_samples_leaf:
                        continue
                    split = self._best_split_for_leaf(X, y_enc, ids)
                    if split is not None:
                        gain, feat, thr, left_idx, right_idx = split
                        candidates.append(
                            (gain, tree_idx, leaf, feat, thr, left_idx, right_idx)
                        )
            if not candidates:
                break
            candidates.sort(key=lambda z: z[0], reverse=True)
            gain, tree_idx, leaf, feat, thr, left_idx, right_idx = candidates[0]
            if gain <= self.min_impurity_decrease:
                break
            leaf.feature = feat
            leaf.threshold = thr
            leaf.left = FIGSNode(
                value=np.array([0.0]),
                sample_idx=left_idx,
                parent_id=leaf.node_id,
                node_id=self._next_node_id()
            )
            leaf.right = FIGSNode(
                value=np.array([0.0]),
                sample_idx=right_idx,
                parent_id=leaf.node_id,
                node_id=self._next_node_id()
            )
            leaf.sample_idx = None
            self.trees_[tree_idx].splits += 1
            for child in (leaf.left, leaf.right):
                child.value[0] = np.mean(residual[child.sample_idx, tree_idx])
            raw = self._predict_raw(X)
            prob = self._softmax(raw)
            self.split_history_.append({
                "gain": float(gain),
                "tree": int(tree_idx),
                "feature": int(feat),
                "threshold": float(thr)
            })
        self.n_splits_ = len(self.split_history_)
        return self

    @property
    def feature_importances_(self):
        if not self.split_history_:
            return np.zeros(self.n_features_in_)
        imp = np.zeros(self.n_features_in_)
        total = 0.0
        for item in self.split_history_:
            imp[item["feature"]] += max(item["gain"], 0.0)
            total += max(item["gain"], 0.0)
        return imp / total if total else imp

    def export_rules(self, feature_names=None):
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{j}" for j in range(self.n_features_in_)]
        )
        out = []
        for k, tree in enumerate(self.trees_):
            def walk(node, path):
                if node.is_leaf:
                    cond = " AND ".join(path) if path else "TRUE"
                    out.append(
                        f"Class={self.classes_[k]} | IF {cond} "
                        f"THEN contribution={node.value[0]:.6f}"
                    )
                    return
                f = names[node.feature]
                walk(node.left, path + [f"{f} <= {node.threshold:.6g}"])
                walk(node.right, path + [f"{f} > {node.threshold:.6g}"])
            walk(tree.root, [])
        return out