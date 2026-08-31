from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass
class GMMClassInfo:
    label: Hashable
    n_components: int
    n_original: int
    n_generated: int
    component_weights: np.ndarray = field(repr=False)
    validation_log_likelihoods: list[float] = field(default_factory=list, repr=False)


class GMMSampling:

    def __init__(
        self,
        minority_classes: Sequence[Hashable] | None = None,
        majority_classes: Sequence[Hashable] | None = None,
        *,
        k_neighbors: int = 5,
        max_components: int = 8,
        validation_size: float = 0.25,
        covariance_type: str = "full",
        reg_covar: float = 1e-5,
        random_state: int = 42,
        scale_features: bool = True,
    ) -> None:
        self.minority_classes = None if minority_classes is None else tuple(minority_classes)
        self.majority_classes = None if majority_classes is None else tuple(majority_classes)
        self.k_neighbors = int(k_neighbors)
        self.max_components = int(max_components)
        self.validation_size = float(validation_size)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.scale_features = bool(scale_features)
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if self.max_components < 1:
            raise ValueError("max_components must be at least 1.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

    @staticmethod
    def _class_similarity(count_a: int, count_b: int) -> float:
        return min(count_a, count_b) / max(count_a, count_b)

    def _resolve_class_roles(self, y: np.ndarray) -> tuple[tuple, tuple]:
        labels, counts = np.unique(y, return_counts=True)
        count_map = dict(zip(labels.tolist(), counts.tolist()))

        if self.minority_classes is not None and self.majority_classes is not None:
            minority = tuple(self.minority_classes)
            majority = tuple(self.majority_classes)
        else:
            mean_count = float(np.mean(counts))
            minority = tuple(label for label, n in count_map.items() if n < mean_count)
            majority = tuple(label for label, n in count_map.items() if n > mean_count)
            if not minority or not majority:
                ordered = sorted(count_map, key=count_map.get)
                majority = (ordered[-1],)
                minority = tuple(ordered[:-1])

        unknown = (set(minority) | set(majority)) - set(labels.tolist())
        if unknown:
            raise ValueError(
                f"Unknown class labels in role configuration: {sorted(unknown, key=str)}"
            )
        if set(minority) & set(majority):
            raise ValueError("A class cannot be both minority and majority.")
        if not minority or not majority:
            raise ValueError("At least one minority and one majority class are required.")
        return minority, majority

    def _safe_levels(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = len(y)
        if n_samples < 2:
            return np.ones(n_samples, dtype=float)

        labels, counts = np.unique(y, return_counts=True)
        count_map = dict(zip(labels.tolist(), counts.tolist()))
        n_query = min(self.k_neighbors + 1, n_samples)
        nn = NearestNeighbors(n_neighbors=n_query, metric="euclidean")
        nn.fit(X)
        neighbour_indices = nn.kneighbors(X, return_distance=False)

        safe = np.zeros(n_samples, dtype=float)
        for i, neighbours in enumerate(neighbour_indices):
            neighbours = neighbours[neighbours != i][:self.k_neighbors]
            if len(neighbours) == 0:
                safe[i] = 1.0
                continue
            own_label = y[i]
            score = 0.0
            for j in neighbours:
                neighbour_label = y[j]
                score += self._class_similarity(count_map[own_label], count_map[neighbour_label])
            safe[i] = score / len(neighbours)
        return np.clip(safe, 0.0, 1.0)

    def _select_components(self, X_class: np.ndarray, seed: int) -> tuple[int, list[float]]:
        max_k = min(self.max_components, max(1, len(X_class) - 1))
        if len(X_class) < 8 or max_k == 1:
            return 1, []

        X_train, X_dev = train_test_split(
            X_class,
            test_size=self.validation_size,
            random_state=seed,
            shuffle=True,
        )
        max_k = min(max_k, len(X_train))
        scores: list[float] = []
        best_k = 1
        best_score = -np.inf

        for k in range(1, max_k + 1):
            model = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                n_init=2,
                max_iter=300,
                random_state=seed + k,
            )
            model.fit(X_train)
            score = float(model.score(X_dev))
            scores.append(score)
            if score > best_score + 1e-8:
                best_score = score
                best_k = k
            elif k > 1:
                break
        return best_k, scores

    @staticmethod
    def _allocate_integer_counts(total: int, weights: np.ndarray) -> np.ndarray:
        if total <= 0:
            return np.zeros(len(weights), dtype=int)
        weights = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        raw = total * weights
        counts = np.floor(raw).astype(int)
        remainder = total - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1
        return counts

    def _sample_component(
        self,
        gmm: GaussianMixture,
        component: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_samples <= 0:
            return np.empty((0, gmm.means_.shape[1]))
        mean = gmm.means_[component]
        if gmm.covariance_type == "full":
            cov = gmm.covariances_[component]
        elif gmm.covariance_type == "tied":
            cov = gmm.covariances_
        elif gmm.covariance_type == "diag":
            cov = np.diag(gmm.covariances_[component])
        elif gmm.covariance_type == "spherical":
            cov = np.eye(len(mean)) * gmm.covariances_[component]
        else:
            raise ValueError(f"Unsupported covariance_type={gmm.covariance_type!r}")
        cov = np.asarray(cov, dtype=float) + np.eye(len(mean)) * self.reg_covar
        return rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

    def _component_log_density(
        self,
        gmm: GaussianMixture,
        X: np.ndarray,
    ) -> np.ndarray:
        columns = []
        for component in range(gmm.n_components):
            mean = gmm.means_[component]
            if gmm.covariance_type == "full":
                cov = gmm.covariances_[component]
            elif gmm.covariance_type == "tied":
                cov = gmm.covariances_
            elif gmm.covariance_type == "diag":
                cov = np.diag(gmm.covariances_[component])
            elif gmm.covariance_type == "spherical":
                cov = np.eye(len(mean)) * gmm.covariances_[component]
            else:
                raise ValueError(
                    f"Unsupported covariance_type={gmm.covariance_type!r}"
                )
            columns.append(
                multivariate_normal.logpdf(
                    X,
                    mean=mean,
                    cov=np.asarray(cov, dtype=float),
                    allow_singular=True,
                )
            )
        return np.column_stack(columns)

    def fit_resample(self, X, y):
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = list(X.columns) if is_dataframe else None
        index_name = X.index.name if is_dataframe else None

        X_arr, y_arr = check_X_y(X, y, accept_sparse=False, dtype=float,
                                 ensure_all_finite=True)
        y_arr = np.asarray(y_arr)
        minority, majority = self._resolve_class_roles(y_arr)
        labels, counts = np.unique(y_arr, return_counts=True)
        class_counts = dict(zip(labels.tolist(), counts.tolist()))

        largest_minority = max(class_counts[c] for c in minority)
        smallest_majority = min(class_counts[c] for c in majority)
        if largest_minority > smallest_majority:
            raise ValueError(
                "Configured minority/majority roles are inconsistent with class "
                "frequencies: a minority class is larger than a majority class."
            )
        balanced_size = int(np.floor((largest_minority + smallest_majority) / 2.0))
        balanced_size = max(1, balanced_size)

        scaler = StandardScaler() if self.scale_features else None
        X_work = scaler.fit_transform(X_arr) if scaler is not None else X_arr.copy()

        safe_levels = self._safe_levels(X_work, y_arr)
        rng = np.random.default_rng(self.random_state)

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        self.class_info_: dict[Hashable, GMMClassInfo] = {}
        self.gmm_models_: dict[Hashable, GaussianMixture] = {}

        role_classes = set(minority) | set(majority)
        for label in labels:
            if label not in role_classes:
                mask = y_arr == label
                X_parts.append(X_work[mask])
                y_parts.append(y_arr[mask])

        for class_offset, label in enumerate(minority):
            mask = y_arr == label
            X_class = X_work[mask]
            safe_class = safe_levels[mask]
            n_to_generate = max(0, balanced_size - len(X_class))
            n_components, dev_scores = self._select_components(
                X_class, self.random_state + 100 * (class_offset + 1)
            )
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                n_init=3,
                max_iter=500,
                random_state=self.random_state + class_offset,
            ).fit(X_class)
            self.gmm_models_[label] = gmm

            component_log_density = self._component_log_density(gmm, X_class)
            unsafe_weight = np.clip(1.0 - safe_class, 0.0, 1.0)
            log_unsafe = np.full_like(unsafe_weight, -np.inf, dtype=float)
            positive = unsafe_weight > 0
            log_unsafe[positive] = np.log(unsafe_weight[positive])
            log_q_tilde = logsumexp(
                log_unsafe[:, None] + component_log_density,
                axis=0,
            )
            if not np.isfinite(log_q_tilde).any():
                q = np.ones(n_components, dtype=float) / n_components
            else:
                q = np.exp(log_q_tilde - logsumexp(log_q_tilde))
            per_component = self._allocate_integer_counts(n_to_generate, q)

            generated = [
                self._sample_component(gmm, k, int(nk), rng)
                for k, nk in enumerate(per_component)
                if nk > 0
            ]
            X_generated = (
                np.vstack(generated) if generated else np.empty(
                    (0, X_work.shape[1]), dtype=float
                )
            )
            X_parts.append(np.vstack([X_class, X_generated]))
            y_parts.append(np.full(len(X_class) + len(X_generated), label,
                                   dtype=y_arr.dtype))
            self.class_info_[label] = GMMClassInfo(
                label=label,
                n_components=n_components,
                n_original=len(X_class),
                n_generated=len(X_generated),
                component_weights=q,
                validation_log_likelihoods=dev_scores,
            )

        for label in majority:
            mask = y_arr == label
            X_class = X_work[mask]
            y_class = y_arr[mask]
            safe_class = safe_levels[mask]
            n_keep = min(len(X_class), balanced_size)
            keep_idx = np.argsort(-safe_class, kind="mergesort")[:n_keep]
            X_parts.append(X_class[keep_idx])
            y_parts.append(y_class[keep_idx])
            self.class_info_[label] = GMMClassInfo(
                label=label,
                n_components=0,
                n_original=len(X_class),
                n_generated=n_keep - len(X_class),
                component_weights=np.array([], dtype=float),
            )

        X_resampled_work = np.vstack(X_parts)
        y_resampled = np.concatenate(y_parts)
        permutation = rng.permutation(len(y_resampled))
        X_resampled_work = X_resampled_work[permutation]
        y_resampled = y_resampled[permutation]
        X_resampled = scaler.inverse_transform(X_resampled_work) if scaler is not None else X_resampled_work

        self.minority_classes_ = minority
        self.majority_classes_ = majority
        self.class_counts_ = class_counts
        self.balanced_size_ = balanced_size
        self.safe_levels_ = safe_levels
        self.scaler_ = scaler
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.asarray(columns, dtype=object) if columns is not None else None

        if is_dataframe:
            X_out = pd.DataFrame(X_resampled, columns=columns)
            X_out.index.name = index_name
            y_name = y.name if isinstance(y, pd.Series) else "target"
            y_out = pd.Series(y_resampled, name=y_name)
            return X_out, y_out
        return X_resampled, y_resampled

    def get_diagnostics(self) -> pd.DataFrame:
        if not hasattr(self, "class_info_"):
            raise RuntimeError("Call fit_resample before requesting diagnostics.")
        rows = []
        for label, info in self.class_info_.items():
            rows.append(
                {
                    "class": label,
                    "role": "minority" if label in self.minority_classes_ else "majority",
                    "original_size": info.n_original,
                    "target_size": self.balanced_size_,
                    "generated_or_removed": info.n_generated,
                    "selected_components": info.n_components,
                    "component_Q": np.round(info.component_weights, 4).tolist(),
                    "dev_log_likelihoods": np.round(info.validation_log_likelihoods, 4).tolist(),
                }
            )
        return pd.DataFrame(rows).sort_values(
            "class", key=lambda s: s.astype(str)
        ).reset_index(drop=True)