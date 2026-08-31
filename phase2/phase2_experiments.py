from __future__ import annotations
import json
import math
import time
import warnings
from pathlib import Path
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from phase1.gmm_sampling import GMMSampling
from figs_model import FIGSClassifier, FIGSRegressor


class _SafeResamplerMixin:
    def fit_resample(self, X, y):
        try:
            return super().fit_resample(X, y)
        except (ValueError, RuntimeError) as e:
            msg = str(e).lower()
            if ("no samples will be generated" in msg or
                "k_neighbors" in msg or
                "n_neighbors" in msg or
                "not any neigbours belong to the majority class" in msg or
                "adasyn is not suited" in msg):
                return X, y
            raise


class SafeSMOTE(_SafeResamplerMixin, SMOTE):
    pass


class SafeBorderlineSMOTE(_SafeResamplerMixin, BorderlineSMOTE):
    pass


class SafeADASYN(_SafeResamplerMixin, ADASYN):
    pass


SEED = 42
CLASS_ORDER = np.array(["H", "D", "A"])

PARAM_DISTRIBUTIONS = {
    "SVC": {
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        "model__kernel": ["rbf", "poly", "sigmoid"],
    },
    "RandomForest": {
        "n_estimators": [100, 150, 200, 300],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_depth": [None, 10, 20, 30],
    },
    "GBM": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 5, 10],
    },
    "FIGS": {
        "max_splits": [4, 6, 8, 10],
        "min_samples_leaf": [5, 10, 15, 20],
        "min_impurity_decrease": [0.0, 1e-8, 1e-6],
    },
    "SVR": {
        "model__C": [0.1, 1.0, 10.0],
        "model__epsilon": [0.05, 0.1, 0.2],
        "model__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    },
    "KernelRidge": {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__gamma": [0.01, 0.1, 1.0],
        "model__kernel": ["rbf", "poly", "sigmoid"],
    },
    "KernelRidgeApprox": {
        "nystroem__n_components": [50, 80, 120],
        "model__alpha": [0.1, 1.0, 10.0],
        "model__gamma": [0.01, 0.1, 1.0],
        "model__kernel": ["rbf", "poly", "sigmoid"],
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 150, 200, 300],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_depth": [None, 10, 20, 30],
    },
    "GBMRegressor": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
    },
    "XGBoostRegressor": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "LightGBMRegressor": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 5, 10],
    },
    "FIGSRegressor": {
        "max_splits": [4, 6, 8, 10],
        "min_samples_leaf": [5, 10, 15, 20],
        "min_impurity_decrease": [0.0, 1e-8, 1e-6],
    },
}

TUNE_NAMES = {
    "SVC": "SVC",
    "RandomForest": "RandomForest",
    "GBM": "GBM",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "FIGS": "FIGS",
    "SVR": "SVR",
    "KernelRidge": "KernelRidge",
    "KernelRidgeApprox": "KernelRidgeApprox",
    "RandomForestRegressor": "RandomForestRegressor",
    "GBMRegressor": "GBMRegressor",
    "XGBoostRegressor": "XGBoostRegressor",
    "LightGBMRegressor": "LightGBMRegressor",
    "FIGSRegressor": "FIGSRegressor",
}


def ece_score(y_true, prob, n_bins=10):
    y = np.asarray(y_true)
    p = np.asarray(prob)
    pred = p.argmax(1)
    conf = p.max(1)
    acc = (CLASS_ORDER[pred] == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        a = acc[mask].mean()
        c = conf[mask].mean()
        ece += mask.mean() * abs(a - c)
        rows.append((lo, hi, mask.sum(), a, c))
    return float(ece), pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "accuracy", "confidence"])


def multiclass_brier(y, p):
    yi = pd.Categorical(y, categories=CLASS_ORDER).codes
    y_one = np.eye(len(CLASS_ORDER))[yi]
    return float(np.mean(np.sum((p - y_one) ** 2, axis=1)))


def rps(y, p):
    yi = pd.Categorical(y, categories=CLASS_ORDER).codes
    y_one = np.eye(3)[yi]
    return float(
        np.mean(
            np.sum(
                (np.cumsum(p, axis=1) - np.cumsum(y_one, axis=1))[:, :2] ** 2,
                axis=1
            ) / 2.0
        )
    )


def classification_metrics(y, p):
    p = np.asarray(p, dtype=float)
    if np.isnan(p).any():
        raise ValueError("Probabilities contain NaN – cannot compute metrics.")
    p = p / p.sum(axis=1, keepdims=True)
    pred = CLASS_ORDER[p.argmax(1)]
    ece, _ = ece_score(y, p)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=CLASS_ORDER, average=None, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(
            pd.Categorical(y, categories=CLASS_ORDER).codes,
            p,
            labels=[0, 1, 2]
        ),
        "brier": multiclass_brier(y, p),
        "rps": rps(y, p),
        "ece": ece,
    }
    for i, cls in enumerate(CLASS_ORDER):
        metrics[f"precision_{cls}"] = precision[i]
        metrics[f"recall_{cls}"] = recall[i]
        metrics[f"f1_{cls}"] = f1[i]
    return metrics


def safe_peak_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return np.nan


def make_demo_snapshots(
    pre: pd.DataFrame,
    random_state: int = 42,
    minutes=(5, 10, 15, 30, 45, 60, 75, 90)
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for r in pre.itertuples(index=False):
        signal = 0.7 * r.diff_form_points + 0.4 * r.diff_form_xg + 0.2 * r.diff_form_shots
        final_margin = int(r.goal_margin)
        for minute in minutes:
            frac = minute / 90.0
            latent = 0.25 * signal * frac + rng.normal(0, 0.85)
            score_diff = int(np.clip(np.round(latent), -3, 3))
            home_goals = max(score_diff, 0)
            away_goals = max(-score_diff, 0)
            rows.append({
                "match_id": r.match_id,
                "kick_off": r.kick_off,
                "snapshot_minute": minute,
                "snapshot_time_seconds": minute * 60,
                "max_event_time_seconds_used": minute * 60,
                "current_home_goals": home_goals,
                "current_away_goals": away_goals,
                "current_score_diff": score_diff,
                "man_advantage_home": float(rng.choice([-1, 0, 0, 0, 1])),
                "live_home_shots": max(0, int(rng.poisson(3.0 + 0.3 * max(signal, 0) + 0.05 * minute))),
                "live_away_shots": max(0, int(rng.poisson(3.0 + 0.3 * max(-signal, 0) + 0.05 * minute))),
                "live_home_xg": max(0.0, float(rng.gamma(2.0, 0.18 + 0.002 * minute))),
                "live_away_xg": max(0.0, float(rng.gamma(2.0, 0.18 + 0.002 * minute))),
                "live_home_passes": max(0, int(rng.poisson(80 + minute))),
                "live_away_passes": max(0, int(rng.poisson(80 + minute))),
                "live_home_pressures": max(0, int(rng.poisson(14 + 0.2 * minute))),
                "live_away_pressures": max(0, int(rng.poisson(14 + 0.2 * minute))),
                "outcome": r.outcome,
                "goal_margin": final_margin,
                "split": r.split,
            })
    snap = pd.DataFrame(rows)
    snap["live_diff_shots"] = snap.live_home_shots - snap.live_away_shots
    snap["live_diff_xg"] = snap.live_home_xg - snap.live_away_xg
    snap["live_diff_passes"] = snap.live_home_passes - snap.live_away_passes
    snap["live_diff_pressures"] = snap.live_home_pressures - snap.live_away_pressures
    snap["live_diff_event_share"] = snap.live_diff_shots / (snap.live_home_shots + snap.live_away_shots + 1)
    snap["prematch_diff_form_points"] = np.repeat(pre.diff_form_points.values, len(minutes))
    snap["prematch_diff_form_xg"] = np.repeat(pre.diff_form_xg.values, len(minutes))
    snap["prematch_diff_form_shots"] = np.repeat(pre.diff_form_shots.values, len(minutes))
    return snap


def feature_columns(df, task):
    exclude = {"match_id", "kick_off", "outcome", "goal_margin", "split"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_models(seed=SEED):
    models = {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "SVC": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, gamma="scale", probability=True, random_state=seed))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=seed,
            n_jobs=1, class_weight="balanced"
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed
        ),
        "XGBoost": __import__("xgboost").XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", random_state=seed, n_jobs=1
        ),
        "LightGBM": __import__("lightgbm").LGBMClassifier(
            n_estimators=120, learning_rate=0.05, num_leaves=15,
            max_depth=-1, random_state=seed, verbose=-1, n_jobs=1
        ),
        "FIGS": FIGSClassifier(max_splits=8, min_samples_leaf=10, random_state=seed),
    }
    regressors = {
        "DummyPrior": DummyRegressor(strategy="mean"),
        "SVR": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(C=1.0, epsilon=0.1, gamma="scale"))
        ]),
        "KernelRidge": Pipeline([
            ("scale", StandardScaler()),
            ("model", KernelRidge(alpha=1.0, kernel="rbf"))
        ]),
        "KernelRidgeApprox": Pipeline([
            ("scale", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", n_components=50, random_state=seed)),
            ("model", KernelRidge(alpha=1.0))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "GBM": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=2,
            random_state=seed, loss="squared_error"
        ),
        "XGBoost": __import__("xgboost").XGBRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=seed, n_jobs=1
        ),
        "LightGBM": __import__("lightgbm").LGBMRegressor(
            n_estimators=120, learning_rate=0.05, num_leaves=15,
            random_state=seed, verbose=-1, n_jobs=1
        ),
        "FIGS": FIGSRegressor(max_splits=8, min_samples_leaf=10, random_state=seed),
    }
    return models, regressors


def tune_and_get_best(base_model, param_dist, X, y, task_type, n_iter=20, cv=3, seed=SEED):
    if task_type in ("C", "L"):
        scoring = "neg_log_loss"
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        n_iter = min(n_iter, 12)
    else:
        scoring = "neg_mean_squared_error"
        cv_splitter = cv
        n_iter = min(n_iter, 8)
    search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_dist, n_iter=n_iter,
        cv=cv_splitter, scoring=scoring, n_jobs=1,
        random_state=seed, refit=True, return_train_score=False
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def margin_to_probabilities(margin_train_oof, y_class_train, margin_test):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y_class_train)
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED)
    lr.fit(margin_train_oof.reshape(-1, 1), y_enc)
    probs = lr.predict_proba(margin_test.reshape(-1, 1))
    class_order_encoded = le.transform(CLASS_ORDER)
    idx = {le.classes_[i]: i for i in range(len(le.classes_))}
    prob_reordered = np.zeros((probs.shape[0], len(CLASS_ORDER)))
    for i, c in enumerate(CLASS_ORDER):
        if c in idx:
            prob_reordered[:, i] = probs[:, idx[c]]
    return prob_reordered


def run_all(pre: pd.DataFrame, snapshots: pd.DataFrame, out: Path, seed=SEED):
    out.mkdir(parents=True, exist_ok=True)
    cls_models, reg_models = build_models(seed)
    results = []
    feature_sets = {
        "C": (pre, "outcome"),
        "R": (pre, "goal_margin"),
        "L": (snapshots, "outcome"),
        "L_reg": (snapshots, "goal_margin"),
    }
    reg_fitted_models = {}
    reg_test_predictions = {}
    reg_test_targets = {}
    reg_test_outcomes = {}
    run_all.margin_train_oof_dict = {}
    run_all.y_class_train = None
    tuning_records = []
    gmm_diagnostics = []

    for task, (frame, target) in feature_sets.items():
        print(f"\n{'=' * 60}")
        print(f"  RUNNING TASK: {task.upper()}")
        print(f"{'=' * 60}")
        train = frame[frame.split == "train"].copy()
        test = frame[frame.split == "test"].copy()
        cols = feature_columns(frame, task)
        cols = [c for c in cols if c in train.columns]
        Xtr = train[cols].fillna(0.0).to_numpy(float)
        Xte = test[cols].fillna(0.0).to_numpy(float)
        odds_available_in_test = False

        if task == "C":
            if "odds_tagged" in test.columns:
                tagged_mask = test["odds_tagged"].astype(bool)
                if tagged_mask.sum() > 0:
                    odds_available_in_test = True
                    test = test.loc[tagged_mask].copy()
                    Xte = test[cols].fillna(0.0).to_numpy(float)
                    print(f"  -> Keeping only {len(test)} odds-tagged matches for evaluation.")
                else:
                    print("  -> No tagged odds in test set. MarketOdds baseline will be skipped.")
            else:
                print("  -> 'odds_tagged' column missing. MarketOdds baseline will be skipped.")

        if task in ("C", "L"):
            ytr_raw = train[target].to_numpy()
            yte = test[target].to_numpy()
            ytr = pd.Categorical(ytr_raw, categories=CLASS_ORDER).codes
            adaptive_k = 1
            min_class_count = np.bincount(ytr).min()
            skip_adasyn = (min_class_count < 2)

            for name, base in cls_models.items():
                if name == "MarketOdds":
                    if not odds_available_in_test:
                        continue
                    if "market_p_H" not in test.columns:
                        continue
                    p = test[["market_p_H", "market_p_D", "market_p_A"]].to_numpy()
                    if np.isnan(p).any():
                        print("  Warning: MarketOdds probabilities contain NaN. Skipping.")
                        continue
                    metrics = classification_metrics(yte, p)
                    results.append({
                        "task": task, "model": name, "calibration": "none",
                        "train_seconds": 0.0, "peak_memory_mb": 0.0, **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p[:, 0],
                        "p_D": p[:, 1],
                        "p_A": p[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_none.csv", index=False)
                    continue

                if name in ("RandomForest_SMOTE", "RandomForest_BorderlineSMOTE", "RandomForest_ADASYN"):
                    if name == "RandomForest_ADASYN" and skip_adasyn:
                        continue
                    if name == "RandomForest_SMOTE":
                        resampler = SafeSMOTE(random_state=seed, k_neighbors=adaptive_k)
                    elif name == "RandomForest_BorderlineSMOTE":
                        resampler = SafeBorderlineSMOTE(random_state=seed, k_neighbors=adaptive_k)
                    else:
                        resampler = SafeADASYN(random_state=seed, n_neighbors=adaptive_k)

                    rf_params = PARAM_DISTRIBUTIONS["RandomForest"]
                    temp_rf = RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
                    tuned_rf, best_params = tune_and_get_best(
                        temp_rf, rf_params, Xtr, ytr, task_type=task, n_iter=12, cv=3, seed=seed
                    )
                    tuning_records.append({
                        "task": task,
                        "model": f"{name}_RF_tuned",
                        "best_params": json.dumps(best_params),
                        "best_score": np.nan
                    })
                    pipeline = make_imb_pipeline(resampler, tuned_rf)
                    model = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3, ensemble=True)
                    t0 = time.perf_counter()
                    model.fit(Xtr, ytr)
                    elapsed = time.perf_counter() - t0
                    p_after = model.predict_proba(Xte)
                    metrics = classification_metrics(yte, p_after)
                    results.append({
                        "task": task, "model": name, "calibration": "sigmoid",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - 0,
                        **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p_after[:, 0],
                        "p_D": p_after[:, 1],
                        "p_A": p_after[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_sigmoid.csv", index=False)
                    if task == "L":
                        mins = np.sort(test.snapshot_minute.unique())
                        for m in mins:
                            mask = test.snapshot_minute.to_numpy() == m
                            mm = classification_metrics(yte[mask], p_after[mask])
                            pd.DataFrame([{
                                "minute": m,
                                "model": name,
                                "calibration": "sigmoid",
                                **mm
                            }]).to_csv(out / f"tmp_minute_{task}_{name}_sigmoid_{m}.csv", index=False)
                    if hasattr(model, "feature_importances_"):
                        fi = np.asarray(model.feature_importances_)
                        pd.DataFrame({"feature": cols, "importance": fi}).sort_values(
                            "importance", ascending=False
                        ).to_csv(out / f"feature_importance_{task}_{name}.csv", index=False)
                    continue

                if name == "RandomForest_GMM":
                    counts = pd.Series(ytr_raw).value_counts()
                    majority_label = counts.idxmax()
                    minority_labels = tuple(c for c in CLASS_ORDER if c != majority_label)
                    min_class_size = min(counts)
                    adaptive_k = min(5, max(1, min_class_size - 1))
                    print(f"  GMM: minority={minority_labels}, majority={majority_label}, k={adaptive_k}")
                    sampler = GMMSampling(
                        minority_classes=minority_labels,
                        majority_classes=(majority_label,),
                        covariance_type='diag',
                        max_components=8,
                        random_state=seed,
                        k_neighbors=adaptive_k,
                        scale_features=True
                    )
                    X_res, y_res = sampler.fit_resample(Xtr, ytr_raw)
                    rf_params = PARAM_DISTRIBUTIONS["RandomForest"]
                    temp_rf = RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
                    tuned_rf, best_params = tune_and_get_best(
                        temp_rf, rf_params, X_res, y_res, task_type=task, n_iter=12, cv=3, seed=seed
                    )
                    tuning_records.append({
                        "task": task,
                        "model": "GMM_RF_tuned",
                        "best_params": json.dumps(best_params),
                        "best_score": np.nan
                    })
                    model = CalibratedClassifierCV(tuned_rf, method='sigmoid', cv=3, ensemble=True)
                    t0 = time.perf_counter()
                    model.fit(X_res, y_res)
                    elapsed = time.perf_counter() - t0
                    p_after = model.predict_proba(Xte)
                    metrics = classification_metrics(yte, p_after)
                    results.append({
                        "task": task, "model": name, "calibration": "sigmoid",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - 0,
                        **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p_after[:, 0],
                        "p_D": p_after[:, 1],
                        "p_A": p_after[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_sigmoid.csv", index=False)
                    if task == "L":
                        mins = np.sort(test.snapshot_minute.unique())
                        for m in mins:
                            mask = test.snapshot_minute.to_numpy() == m
                            mm = classification_metrics(yte[mask], p_after[mask])
                            pd.DataFrame([{
                                "minute": m,
                                "model": name,
                                "calibration": "sigmoid",
                                **mm
                            }]).to_csv(out / f"tmp_minute_{task}_{name}_sigmoid_{m}.csv", index=False)
                    if hasattr(model, "feature_importances_"):
                        fi = np.asarray(model.feature_importances_)
                        pd.DataFrame({"feature": cols, "importance": fi}).sort_values(
                            "importance", ascending=False
                        ).to_csv(out / f"feature_importance_{task}_{name}.csv", index=False)
                    diag_df = sampler.get_diagnostics()
                    diag_df.to_csv(out / f"gmm_diagnostics_{task}_{name}.csv", index=False)
                    gmm_diagnostics.append(diag_df.assign(task=task, model=name))
                    continue

                if name == "DummyPrior":
                    model = clone(base)
                    t0 = time.perf_counter()
                    model.fit(Xtr, ytr)
                    elapsed = time.perf_counter() - t0
                    p = model.predict_proba(Xte)
                    metrics = classification_metrics(yte, p)
                    results.append({
                        "task": task, "model": name, "calibration": "none",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - 0,
                        **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p[:, 0],
                        "p_D": p[:, 1],
                        "p_A": p[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_none.csv", index=False)
                    continue

                base_model = clone(base)
                tune_key = TUNE_NAMES.get(name)
                if tune_key in PARAM_DISTRIBUTIONS:
                    param_dist = PARAM_DISTRIBUTIONS[tune_key]
                    best_estimator, best_params = tune_and_get_best(
                        base_model, param_dist, Xtr, ytr, task_type=task, n_iter=12, cv=3, seed=seed
                    )
                    tuning_records.append({
                        "task": task,
                        "model": name,
                        "best_params": json.dumps(best_params),
                        "best_score": getattr(best_estimator, "best_score_", np.nan)
                    })
                    base_model = best_estimator
                else:
                    base_model.fit(Xtr, ytr)

                if hasattr(base_model, "predict_proba"):
                    p_none = base_model.predict_proba(Xte)
                else:
                    p_none = None
                if p_none is not None:
                    metrics = classification_metrics(yte, p_none)
                    results.append({
                        "task": task, "model": name, "calibration": "none",
                        "train_seconds": 0.0, "peak_memory_mb": 0.0, **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p_none[:, 0],
                        "p_D": p_none[:, 1],
                        "p_A": p_none[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_none.csv", index=False)

                try:
                    cal_sigmoid = CalibratedClassifierCV(base_model, method='sigmoid', cv=3, ensemble=True)
                    t0 = time.perf_counter()
                    cal_sigmoid.fit(Xtr, ytr)
                    elapsed = time.perf_counter() - t0
                    p_sigmoid = cal_sigmoid.predict_proba(Xte)
                    metrics = classification_metrics(yte, p_sigmoid)
                    results.append({
                        "task": task, "model": name, "calibration": "sigmoid",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - 0,
                        **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p_sigmoid[:, 0],
                        "p_D": p_sigmoid[:, 1],
                        "p_A": p_sigmoid[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_sigmoid.csv", index=False)
                except Exception as e:
                    print(f"  Sigmoid calibration failed for {name}: {e}")

                try:
                    cal_iso = CalibratedClassifierCV(base_model, method='isotonic', cv=3, ensemble=True)
                    t0 = time.perf_counter()
                    cal_iso.fit(Xtr, ytr)
                    elapsed = time.perf_counter() - t0
                    p_iso = cal_iso.predict_proba(Xte)
                    metrics = classification_metrics(yte, p_iso)
                    results.append({
                        "task": task, "model": name, "calibration": "isotonic",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - 0,
                        **metrics
                    })
                    pd.DataFrame({
                        "y_true": yte,
                        "p_H": p_iso[:, 0],
                        "p_D": p_iso[:, 1],
                        "p_A": p_iso[:, 2]
                    }).to_csv(out / f"predictions_{task}_{name}_isotonic.csv", index=False)
                except Exception as e:
                    print(f"  Isotonic calibration failed for {name}: {e}")

                if hasattr(base_model, "feature_importances_"):
                    fi = np.asarray(base_model.feature_importances_)
                    pd.DataFrame({"feature": cols, "importance": fi}).sort_values(
                        "importance", ascending=False
                    ).to_csv(out / f"feature_importance_{task}_{name}.csv", index=False)

            if task == "L":
                first_snap_idx = test.groupby('match_id')['snapshot_minute'].idxmin()
                first_snap = test.loc[first_snap_idx]
                if "market_p_H" in first_snap.columns and first_snap["odds_tagged"].astype(bool).any():
                    p_market = first_snap[["market_p_H", "market_p_D", "market_p_A"]].to_numpy()
                    y_market = first_snap["outcome"].to_numpy()
                    metrics_market = classification_metrics(y_market, p_market)
                    results.append({
                        "task": "L0", "model": "MarketOdds_L0", "calibration": "none",
                        "train_seconds": 0.0, "peak_memory_mb": 0.0, **metrics_market
                    })
                    pd.DataFrame({
                        "match_id": first_snap["match_id"],
                        "snapshot_minute": first_snap["snapshot_minute"],
                        "y_true": y_market,
                        "p_H": p_market[:, 0],
                        "p_D": p_market[:, 1],
                        "p_A": p_market[:, 2]
                    }).to_csv(out / "predictions_L0_MarketOdds_none.csv", index=False)

            try:
                del Xtr, Xte, ytr_raw, yte, ytr
            except UnboundLocalError:
                pass
            gc.collect()

        else:
            ytr = train[target].to_numpy(float)
            yte = test[target].to_numpy(float)
            if task == "R":
                y_class_test = test["outcome"].to_numpy()
                y_class_train = train["outcome"].to_numpy()
                run_all.y_class_train = y_class_train

            for name, base in reg_models.items():
                try:
                    print(f"  Training regression model: {name}...")
                    sys.stdout.flush()
                    t0 = time.perf_counter()
                    mem0 = safe_peak_memory_mb()
                    model = clone(base)
                    if name != "DummyPrior":
                        tune_key = TUNE_NAMES.get(name)
                        if tune_key in PARAM_DISTRIBUTIONS:
                            param_dist = PARAM_DISTRIBUTIONS[tune_key]
                            best_model, best_params = tune_and_get_best(
                                model, param_dist, Xtr, ytr, task_type=task, n_iter=8, cv=3, seed=seed
                            )
                            model = best_model
                            tuning_records.append({
                                "task": task,
                                "model": name,
                                "best_params": json.dumps(best_params),
                                "best_score": getattr(best_model, "best_score_", np.nan)
                            })
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    elapsed = time.perf_counter() - t0
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr_val = np.corrcoef(yte, pred)[0, 1]
                    corr_val = 0.0 if np.isnan(corr_val) else float(corr_val)
                    results.append({
                        "task": task,
                        "model": name,
                        "calibration": "none",
                        "train_seconds": elapsed,
                        "peak_memory_mb": safe_peak_memory_mb() - mem0,
                        "mae": mean_absolute_error(yte, pred),
                        "rmse": math.sqrt(mean_squared_error(yte, pred)),
                        "r2": r2_score(yte, pred),
                        "corr": corr_val
                    })
                    pd.DataFrame({"y_true": yte, "prediction": pred}).to_csv(
                        out / f"predictions_{task}_{name}_none.csv", index=False
                    )
                    if hasattr(model, "feature_importances_"):
                        fi = np.asarray(model.feature_importances_)
                        pd.DataFrame({"feature": cols, "importance": fi}).sort_values(
                            "importance", ascending=False
                        ).to_csv(out / f"feature_importance_{task}_{name}.csv", index=False)
                    if task == "L_reg":
                        mins = np.sort(test.snapshot_minute.unique())
                        for m in mins:
                            mask = test.snapshot_minute.to_numpy() == m
                            yte_m = yte[mask]
                            pred_m = pred[mask]
                            if len(yte_m) == 0:
                                continue
                            with np.errstate(invalid='ignore', divide='ignore'):
                                corr_val_m = np.corrcoef(yte_m, pred_m)[0, 1]
                            corr_val_m = 0.0 if np.isnan(corr_val_m) else float(corr_val_m)
                            pd.DataFrame([{
                                "minute": m,
                                "model": name,
                                "calibration": "none",
                                "mae": mean_absolute_error(yte_m, pred_m),
                                "rmse": math.sqrt(mean_squared_error(yte_m, pred_m)),
                                "corr": corr_val_m
                            }]).to_csv(out / f"tmp_minute_{task}_{name}_none_{m}.csv", index=False)
                    if task == "R":
                        reg_fitted_models[name] = model
                        reg_test_predictions[name] = pred
                        reg_test_targets[name] = yte
                        reg_test_outcomes[name] = y_class_test
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                        from sklearn.base import clone as sk_clone
                        oof_predictions = cross_val_predict(
                            sk_clone(model), Xtr, ytr, cv=cv, method='predict'
                        )
                        run_all.margin_train_oof_dict[name] = oof_predictions

                    try:
                        del pred, model
                    except UnboundLocalError:
                        pass
                    gc.collect()
                except Exception as e:
                    print(f"  ERROR: {name} failed: {repr(e)}")
                    continue

            try:
                del Xtr, Xte, ytr, yte
            except UnboundLocalError:
                pass
            if task == "R":
                try:
                    del y_class_test, y_class_train
                except UnboundLocalError:
                    pass
            gc.collect()

    if tuning_records:
        pd.DataFrame(tuning_records).to_csv(out / "hyperparameter_tuning_results.csv", index=False)
    if gmm_diagnostics:
        combined_gmm = pd.concat(gmm_diagnostics, ignore_index=True)
        combined_gmm.to_csv(out / "gmm_diagnostics_all.csv", index=False)
    if reg_fitted_models:
        for name, margin_test_pred in reg_test_predictions.items():
            if name in run_all.margin_train_oof_dict and run_all.y_class_train is not None:
                margin_train_oof = run_all.margin_train_oof_dict[name]
                y_class_train = run_all.y_class_train
                probs_test = margin_to_probabilities(
                    margin_train_oof, y_class_train, margin_test_pred
                )
                y_class_test = reg_test_outcomes[name]
                metrics = classification_metrics(y_class_test, probs_test)
                results.append({
                    "task": "R_to_C",
                    "model": f"Margin_to_Class_{name}",
                    "calibration": "none",
                    "train_seconds": 0.0,
                    "peak_memory_mb": 0.0,
                    **metrics
                })
                pd.DataFrame({
                    "y_true": y_class_test,
                    "p_H": probs_test[:, 0],
                    "p_D": probs_test[:, 1],
                    "p_A": probs_test[:, 2]
                }).to_csv(out / f"predictions_R_to_C_{name}_none.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "phase2_results_all.csv", index=False)
    return results_df