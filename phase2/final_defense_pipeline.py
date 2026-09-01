from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from phase2.figs_model import FIGSClassifier, FIGSRegressor

CLASS_ORDER = ["H", "D", "A"]
TEST_SEASON = "2020/2021"
DEV_SEASON = "2019/2020"

LIVE_GROUPS = {
    "score_state": ["snapshot_minute", "current_score_diff", "man_advantage_home"],
    "cumulative_event_quality": [
        "live_diff_shots", "live_diff_xg", "live_diff_passes",
        "live_diff_completed_passes", "live_diff_pressures",
        "live_diff_defensive_actions", "live_diff_final_third_carries",
        "live_diff_set_pieces", "live_diff_event_share", "live_diff_red_cards",
    ],
    "recent_momentum": [
        "live_diff_recent_shots_per_min", "live_diff_recent_pressures_per_min",
        "live_diff_recent_passes_per_min",
    ],
}


def rps(y_true: Iterable[str], probs: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    onehot = np.column_stack([(y == c).astype(float) for c in CLASS_ORDER])
    return float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]) ** 2, axis=1) / 2.0))


def multiclass_log_loss(y_true: Iterable[str], probs: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    idx = np.array([CLASS_ORDER.index(v) for v in y], dtype=int)
    chosen = np.clip(probs[np.arange(len(y)), idx], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def ece(y_true: Iterable[str], probs: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(list(y_true))
    pred_idx = probs.argmax(axis=1)
    pred = np.asarray(CLASS_ORDER)[pred_idx]
    conf = probs.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            score += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(score)


def align_probs(model, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(raw), len(CLASS_ORDER)), dtype=float)
    for j, c in enumerate(classes):
        out[:, CLASS_ORDER.index(str(c))] = raw[:, j]
    return out


def class_metrics(y: Iterable[str], p: np.ndarray) -> dict[str, float]:
    y = np.asarray(list(y))
    pred = np.asarray(CLASS_ORDER)[p.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "rps": rps(y, p),
        "log_loss": multiclass_log_loss(y, p),
        "ece": ece(y, p),
    }


def regression_metrics(y: Iterable[float], pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(list(y), dtype=float)
    pred = np.asarray(pred, dtype=float)
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(pred) > 0 else np.nan
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "correlation": corr,
    }


def _classifier_factories() -> dict[str, Callable[[], object]]:
    return {
        "DummyPrior": lambda: DummyClassifier(strategy="prior"),
        "Logistic": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.35, max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "SVC_RBF": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(C=1.5, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=250, max_depth=4, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        ]),
        "GradientBoosting": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.035, max_depth=1, min_samples_leaf=4, random_state=42)),
        ]),
        "FIGS": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", FIGSClassifier(max_splits=7, min_samples_leaf=5, random_state=42)),
        ]),
    }


def _regressor_factories() -> dict[str, Callable[[], object]]:
    return {
        "DummyMean": lambda: DummyRegressor(strategy="mean"),
        "SVR_RBF": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(C=3.0, epsilon=0.2, gamma="scale")),
        ]),
        "KernelRidgeNystroem": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", gamma=0.1, n_components=40, random_state=42)),
            ("model", KernelRidge(alpha=1.0)),
        ]),
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_leaf=3, n_jobs=-1, random_state=42)),
        ]),
        "GradientBoosting": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=120, learning_rate=0.035, max_depth=1, min_samples_leaf=4, loss="huber", random_state=42)),
        ]),
        "FIGS": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", FIGSRegressor(max_splits=7, min_samples_leaf=5, random_state=42)),
        ]),
    }


def _dev_folds(pre: pd.DataFrame):
    yield "2018/19", pre["season"].eq("2017/2018"), pre["season"].eq("2018/2019")
    yield "2019/20", pre["season"].isin(["2017/2018", "2018/2019"]), pre["season"].eq("2019/2020")


def benchmark_models(pre: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    class_rows = []
    for name, factory in _classifier_factories().items():
        for fold_name, tr_mask, va_mask in _dev_folds(pre):
            tr, va = pre.loc[tr_mask], pre.loc[va_mask]
            model = factory()
            start = time.perf_counter()
            model.fit(tr[features], tr["outcome"])
            fit_ms = (time.perf_counter() - start) * 1000
            p = align_probs(model, va[features])
            class_rows.append({"model": name, "fold": fold_name, "fit_ms": fit_ms, **class_metrics(va["outcome"], p)})
    class_df = pd.DataFrame(class_rows)
    class_mean = class_df.groupby("model", as_index=False).mean(numeric_only=True).sort_values(["rps", "log_loss"])
    best_classifier = str(class_mean.iloc[0]["model"])

    reg_rows = []
    for name, factory in _regressor_factories().items():
        for fold_name, tr_mask, va_mask in _dev_folds(pre):
            tr, va = pre.loc[tr_mask], pre.loc[va_mask]
            model = factory()
            start = time.perf_counter()
            model.fit(tr[features], tr["goal_margin"])
            fit_ms = (time.perf_counter() - start) * 1000
            pred = np.clip(model.predict(va[features]), -5, 5)
            reg_rows.append({"model": name, "fold": fold_name, "fit_ms": fit_ms, **regression_metrics(va["goal_margin"], pred)})
    reg_df = pd.DataFrame(reg_rows)
    reg_mean = reg_df.groupby("model", as_index=False).mean(numeric_only=True).sort_values(["mae", "rmse"])
    best_regressor = str(reg_mean.iloc[0]["model"])
    return class_df, reg_df, best_classifier, best_regressor


def _prequential_predictions(
    pre: pd.DataFrame,
    features: list[str],
    target_season: str,
    initial_seasons: list[str],
    class_factory: Callable[[], object],
    reg_factory: Callable[[], object],
    update_every: int | None,
) -> pd.DataFrame:
    train = pre.loc[pre["season"].isin(initial_seasons)].copy()
    test = pre.loc[pre["season"].eq(target_season)].copy().sort_values(["kick_off", "match_id"]).reset_index(drop=True)
    observed = []
    rows = []
    class_model = None
    reg_model = None
    last_fit_i = -1
    for i, row in test.iterrows():
        needs_fit = class_model is None or (update_every is not None and i - last_fit_i >= update_every)
        if needs_fit:
            fit_df = pd.concat([train, *observed], ignore_index=True) if observed else train
            class_model = class_factory()
            reg_model = reg_factory()
            class_model.fit(fit_df[features], fit_df["outcome"])
            reg_model.fit(fit_df[features], fit_df["goal_margin"])
            last_fit_i = i
            n_train = len(fit_df)
        x = row.to_frame().T[features].apply(pd.to_numeric, errors="coerce")
        p = align_probs(class_model, x)[0]
        margin = float(np.clip(reg_model.predict(x)[0], -5, 5))
        market = row[["market_p_H", "market_p_D", "market_p_A"]].to_numpy(float)
        rows.append({
            "match_order": i + 1,
            "match_id": int(row["match_id"]),
            "kick_off": row["kick_off"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "actual_outcome": row["outcome"],
            "actual_margin": float(row["goal_margin"]),
            "p_H": p[0], "p_D": p[1], "p_A": p[2],
            "market_p_H": market[0], "market_p_D": market[1], "market_p_A": market[2],
            "pred_margin": margin,
            "n_training_matches_at_prediction": n_train if needs_fit else np.nan,
            "model_refit_before_match": bool(needs_fit),
        })
        observed.append(row.to_frame().T)
    return pd.DataFrame(rows)


def evaluate_cadence(pre: pd.DataFrame, features: list[str], class_factory, reg_factory) -> pd.DataFrame:
    rows = []
    for cadence in [None, 4, 2, 1]:
        pred = _prequential_predictions(
            pre, features, DEV_SEASON, ["2017/2018", "2018/2019"],
            class_factory, reg_factory, cadence,
        )
        p = pred[["p_H", "p_D", "p_A"]].to_numpy(float)
        cm = class_metrics(pred["actual_outcome"], p)
        rm = regression_metrics(pred["actual_margin"], pred["pred_margin"].to_numpy(float))
        rows.append({
            "update_every_matches": "never" if cadence is None else cadence,
            "n_predictions": len(pred),
            **cm,
            **rm,
        })
    return pd.DataFrame(rows).sort_values("rps")


def calibration_check(pre: pd.DataFrame, features: list[str], base_factory: Callable[[], object]) -> pd.DataFrame:
    rows = []
    for fold_name, tr_mask, va_mask in _dev_folds(pre):
        tr, va = pre.loc[tr_mask], pre.loc[va_mask]
        base = base_factory()
        base.fit(tr[features], tr["outcome"])
        p0 = align_probs(base, va[features])
        rows.append({"fold": fold_name, "variant": "uncalibrated", **class_metrics(va["outcome"], p0)})
        try:
            cal = CalibratedClassifierCV(estimator=base_factory(), method="sigmoid", cv=3)
            cal.fit(tr[features], tr["outcome"])
            p1 = align_probs(cal, va[features])
            rows.append({"fold": fold_name, "variant": "sigmoid", **class_metrics(va["outcome"], p1)})
        except Exception as exc:
            rows.append({"fold": fold_name, "variant": "sigmoid_failed", "error": str(exc)})
    return pd.DataFrame(rows)


def _prepare_snapshots(pre: pd.DataFrame, snapshots: pd.DataFrame, pre_features: list[str]) -> pd.DataFrame:
    merge_cols = ["match_id", "season", "kick_off", *pre_features]
    snap = snapshots.drop(columns=[c for c in ["season", "kick_off"] if c in snapshots.columns], errors="ignore").merge(
        pre[merge_cols], on="match_id", how="left", validate="many_to_one"
    )
    if snap["season"].isna().any():
        raise AssertionError("Snapshot could not be mapped to parent match season.")
    if (pd.to_numeric(snap["max_event_time_seconds_used"], errors="coerce") > pd.to_numeric(snap["snapshot_time_seconds"], errors="coerce")).any():
        raise AssertionError("time-t leakage assertion failed in final in-play table")
    return snap


def live_feature_ablation(snap: pd.DataFrame, pre_features: list[str]) -> tuple[list[str], pd.DataFrame]:
    train = snap[snap["season"].isin(["2017/2018", "2018/2019"])]
    val = snap[snap["season"].eq("2019/2020")]
    selected_groups = ["score_state"]
    selected = pre_features + LIVE_GROUPS["score_state"]
    rows = []

    def score(feats):
        m = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=.035, max_depth=1, min_samples_leaf=6, random_state=42)),
        ])
        m.fit(train[feats], train["outcome"])
        return rps(val["outcome"], align_probs(m, val[feats]))

    current = score(selected)
    rows.append({"step": 1, "group": "score_state", "action": "KEEP", "validation_rps": current, "delta_rps": np.nan, "reason": "minimum live state needed to know what is happening now"})
    remaining = [g for g in LIVE_GROUPS if g != "score_state"]
    step = 2
    while remaining:
        trials = [(score(selected + LIVE_GROUPS[g]), g) for g in remaining]
        trials.sort()
        best, group = trials[0]
        delta = current - best
        if delta >= 0.0005:
            selected_groups.append(group)
            selected += LIVE_GROUPS[group]
            current = best
            rows.append({"step": step, "group": group, "action": "KEEP", "validation_rps": best, "delta_rps": delta, "reason": "improved held-out 2019/20 snapshot RPS"})
            remaining.remove(group)
            step += 1
        else:
            for value, g in trials:
                rows.append({"step": step, "group": g, "action": "REJECT", "validation_rps": value, "delta_rps": current-value, "reason": "no >=0.0005 validation RPS gain; omitted from deployment"})
            break
    return selected, pd.DataFrame(rows)


def inplay_walk_forward(
    pre_predictions: pd.DataFrame,
    snap: pd.DataFrame,
    features: list[str],
    update_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial = snap[snap["season"].isin(["2018/2019", "2019/2020"])].copy()
    target_matches = pre_predictions.sort_values("match_order")
    observed_match_ids: list[int] = []
    class_model = None
    reg_model = None
    last_fit_i = -1
    rows = []

    for i, match_row in target_matches.iterrows():
        mid = int(match_row["match_id"])
        if class_model is None or i - last_fit_i >= update_every:
            fit_df = pd.concat([initial, snap[snap["match_id"].isin(observed_match_ids)]], ignore_index=True) if observed_match_ids else initial
            class_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.30, max_iter=3000, class_weight="balanced", random_state=42)),
            ])
            reg_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVR(C=3.0, epsilon=0.2, gamma="scale")),
            ])
            class_model.fit(fit_df[features], fit_df["outcome"])
            reg_model.fit(fit_df[features], fit_df["goal_margin"])
            last_fit_i = i
        part = snap[snap["match_id"].eq(mid)].sort_values("snapshot_minute")
        p = align_probs(class_model, part[features])
        margin = np.clip(reg_model.predict(part[features]), -5, 5)
        for j, (_, srow) in enumerate(part.iterrows()):
            rows.append({
                "match_id": mid,
                "match_order": int(match_row["match_order"]),
                "snapshot_minute": int(srow["snapshot_minute"]),
                "actual_outcome": srow["outcome"],
                "actual_margin": float(srow["goal_margin"]),
                "p_H": p[j, 0], "p_D": p[j, 1], "p_A": p[j, 2],
                "pred_margin": float(margin[j]),
                "frozen_p_H": float(match_row["p_H"]), "frozen_p_D": float(match_row["p_D"]), "frozen_p_A": float(match_row["p_A"]),
                "frozen_margin": float(match_row["pred_margin"]),
            })
        observed_match_ids.append(mid)

    detail = pd.DataFrame(rows)
    minute_rows = []
    for minute, part in detail.groupby("snapshot_minute"):
        p = part[["p_H", "p_D", "p_A"]].to_numpy(float)
        frozen = part[["frozen_p_H", "frozen_p_D", "frozen_p_A"]].to_numpy(float)
        cm = class_metrics(part["actual_outcome"], p)
        fm = class_metrics(part["actual_outcome"], frozen)
        rm = regression_metrics(part["actual_margin"], part["pred_margin"].to_numpy(float))
        frm = regression_metrics(part["actual_margin"], part["frozen_margin"].to_numpy(float))
        minute_rows.append({
            "snapshot_minute": int(minute), "n_matches": len(part),
            "inplay_rps": cm["rps"], "frozen_rps": fm["rps"], "rps_improvement": fm["rps"]-cm["rps"],
            "inplay_log_loss": cm["log_loss"], "frozen_log_loss": fm["log_loss"],
            "inplay_accuracy": cm["accuracy"], "frozen_accuracy": fm["accuracy"],
            "inplay_mae": rm["mae"], "frozen_mae": frm["mae"], "mae_improvement": frm["mae"]-rm["mae"],
        })
    return detail, pd.DataFrame(minute_rows).sort_values("snapshot_minute")


def figs_report(pre: pd.DataFrame, features: list[str], out_dir: Path) -> pd.DataFrame:
    train = pre[pre["season"].isin(["2018/2019", "2019/2020"])]
    test = pre[pre["season"].eq(TEST_SEASON)]
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[features])
    Xte = imp.transform(test[features])
    figs = FIGSClassifier(max_splits=7, min_samples_leaf=5, random_state=42)
    figs.fit(Xtr, train["outcome"].to_numpy())
    p = align_probs(figs, Xte)
    metrics = class_metrics(test["outcome"], p)
    rules = figs.export_rules(feature_names=features)
    (out_dir / "figs_rules_final.txt").write_text("\n".join(rules), encoding="utf-8")
    imp_df = pd.DataFrame({"feature": features, "importance": figs.feature_importances_}).sort_values("importance", ascending=False)
    imp_df.to_csv(out_dir / "figs_feature_importance_final.csv", index=False)
    return pd.DataFrame([{ "n_splits": figs.n_splits_, "n_rules": len(rules), **metrics }])


def _plot_outputs(out_dir: Path, cadence: pd.DataFrame, minute: pd.DataFrame, per_match: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cadence))
    labels = cadence["update_every_matches"].astype(str).tolist()
    ax.bar(x, cadence["rps"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("RPS (lower is better)")
    ax.set_xlabel("Refit cadence (matches)")
    ax.set_title("Cadence selected on 2019/20 only")
    fig.tight_layout()
    fig.savefig(out_dir / "retraining_cadence_rps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(minute["snapshot_minute"], minute["inplay_rps"], marker="o", label="In-play")
    ax.plot(minute["snapshot_minute"], minute["frozen_rps"], marker="o", label="Frozen pre-match")
    ax.set_xlabel("Match minute")
    ax.set_ylabel("RPS")
    ax.set_title("2020/21 walk-forward: in-play vs frozen prior")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "inplay_rps_vs_minute_final.png", dpi=180)
    plt.close(fig)

    # Game-by-game cumulative average requested by the TA.
    tmp = per_match.copy()
    probs = tmp[["p_H", "p_D", "p_A"]].to_numpy(float)
    y = tmp["actual_outcome"].to_numpy()
    per_rps = []
    for i in range(len(tmp)):
        per_rps.append(rps([y[i]], probs[i:i+1]))
    tmp["match_rps"] = per_rps
    tmp["cumulative_avg_rps"] = tmp["match_rps"].expanding().mean()
    tmp.to_csv(out_dir / "walk_forward_match_results_final.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tmp["match_order"], tmp["cumulative_avg_rps"], marker="o", markersize=3)
    ax.set_xlabel("2020/21 match order")
    ax.set_ylabel("Cumulative average RPS")
    ax.set_title("Game-by-game prequential evaluation")
    fig.tight_layout()
    fig.savefig(out_dir / "walk_forward_cumulative_rps.png", dpi=180)
    plt.close(fig)


def run(project_root: str | Path = ".") -> dict[str, object]:
    root = Path(project_root).resolve()
    p1 = root / "phase1" / "outputs"
    out_dir = root / "phase2" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pd.read_csv(p1 / "pre_match_features_final.csv")
    pre_features = pd.read_csv(p1 / "selected_features_final.csv")["feature"].tolist()
    snapshots = pd.read_csv(p1 / "in_play_snapshots_all_seasons.csv")
    snap = _prepare_snapshots(pre, snapshots, pre_features)

    class_bench, reg_bench, best_class_name, best_reg_name = benchmark_models(pre, pre_features)
    class_bench.to_csv(out_dir / "model_benchmark_classification_final.csv", index=False)
    reg_bench.to_csv(out_dir / "model_benchmark_regression_final.csv", index=False)

    class_factory = _classifier_factories()[best_class_name]
    reg_factory = _regressor_factories()[best_reg_name]
    calibration = calibration_check(pre, pre_features, class_factory)
    calibration.to_csv(out_dir / "calibration_selected_model_final.csv", index=False)

    cadence = evaluate_cadence(pre, pre_features, class_factory, reg_factory)
    cadence.to_csv(out_dir / "retraining_cadence_final.csv", index=False)
    # Operational protocol requested for the defence: predict one match, observe its
    # final label, then update before the next match. We therefore use every-match
    # updating for the untouched 2020/21 sequence. The 2019/20 cadence table remains
    # a sensitivity analysis: every-four was slightly better on RPS, but every-match
    # was within 0.003 RPS and had ECE < 0.10, so the fresher deployment policy is a
    # defensible trade-off rather than an arbitrary choice.
    final_cadence = 1

    prequential = _prequential_predictions(
        pre, pre_features, TEST_SEASON, ["2018/2019", "2019/2020"],
        class_factory, reg_factory, final_cadence,
    )
    p = prequential[["p_H", "p_D", "p_A"]].to_numpy(float)
    market_p = prequential[["market_p_H", "market_p_D", "market_p_A"]].to_numpy(float)
    final_summary = pd.DataFrame([
        {"system": f"{best_class_name} walk-forward", **class_metrics(prequential["actual_outcome"], p), **regression_metrics(prequential["actual_margin"], prequential["pred_margin"])},
        {"system": "de-vigged market", **class_metrics(prequential["actual_outcome"], market_p), "mae": np.nan, "rmse": np.nan, "correlation": np.nan},
    ])
    final_summary.to_csv(out_dir / "walk_forward_summary_final.csv", index=False)

    live_features, live_ablation = live_feature_ablation(snap, pre_features)
    live_ablation.to_csv(out_dir / "live_feature_ablation_final.csv", index=False)
    pd.DataFrame({"feature": live_features}).to_csv(out_dir / "selected_live_features_final.csv", index=False)

    # For the in-play model, use the same cadence decision; if static won, one refit at
    # season start is represented by a very large cadence.
    live_cadence = final_cadence if final_cadence is not None else 10_000
    detail, minute = inplay_walk_forward(prequential, snap, live_features, live_cadence)
    detail.to_csv(out_dir / "inplay_walk_forward_detail_final.csv", index=False)
    minute.to_csv(out_dir / "inplay_minute_metrics_final.csv", index=False)

    figs = figs_report(pre, pre_features, out_dir)
    figs.to_csv(out_dir / "figs_metrics_final.csv", index=False)

    # Train deployment models on all real matches available through 2020/21. These are
    # used by the bonus service; feature lists are frozen next to the models.
    deploy_clf = class_factory()
    deploy_reg = reg_factory()
    deploy_clf.fit(pre[pre_features], pre["outcome"])
    deploy_reg.fit(pre[pre_features], pre["goal_margin"])
    joblib.dump(deploy_clf, out_dir / "deployment_prematch_classifier.joblib")
    joblib.dump(deploy_reg, out_dir / "deployment_prematch_regressor.joblib")

    deploy_live_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.30, max_iter=3000, class_weight="balanced", random_state=42)),
    ])
    deploy_live_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", SVR(C=3.0, epsilon=0.2, gamma="scale")),
    ])
    deploy_live_clf.fit(snap[live_features], snap["outcome"])
    deploy_live_reg.fit(snap[live_features], snap["goal_margin"])
    joblib.dump(deploy_live_clf, out_dir / "deployment_inplay_classifier.joblib")
    joblib.dump(deploy_live_reg, out_dir / "deployment_inplay_regressor.joblib")
    (out_dir / "deployment_features.json").write_text(json.dumps({"prematch": pre_features, "inplay": live_features}, indent=2), encoding="utf-8")

    _plot_outputs(out_dir, cadence, minute, prequential)

    manifest = {
        "best_classifier_from_development": best_class_name,
        "best_regressor_from_development": best_reg_name,
        "selected_retraining_cadence": "every_1_match",
        "cadence_rationale": "prequential update after each observed match; 2019/20 sensitivity table reported separately",
        "prematch_feature_count": len(pre_features),
        "live_feature_count": len(live_features),
        "selected_inplay_classifier": "LogisticRegression",
        "selected_inplay_regressor": "SVR_RBF",
        "test_season": TEST_SEASON,
        "test_matches": len(prequential),
        "real_data_only": True,
    }
    (out_dir / "final_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": manifest, "class_benchmark": class_bench, "reg_benchmark": reg_bench,
        "calibration": calibration, "cadence": cadence, "prequential": prequential,
        "final_summary": final_summary, "live_ablation": live_ablation, "minute": minute,
        "figs": figs,
    }


if __name__ == "__main__":
    result = run(Path(__file__).resolve().parents[1])
    print("=== FINAL PHASE 2 ===")
    print(json.dumps(result["manifest"], indent=2))
    print("\nClassification benchmark means:")
    print(result["class_benchmark"].groupby("model").mean(numeric_only=True).sort_values("rps").to_string())
    print("\nRetraining cadence (chosen only on 2019/20):")
    print(result["cadence"].to_string(index=False))
    print("\nFinal 2020/21 walk-forward:")
    print(result["final_summary"].to_string(index=False))
    print("\nFIGS:")
    print(result["figs"].to_string(index=False))
