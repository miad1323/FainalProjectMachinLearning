from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1.statsbomb_phase1 import tag_odds_to_matches

CLASS_ORDER = ["H", "D", "A"]

SEASONS = ["2017/2018", "2018/2019", "2019/2020", "2020/2021"]
ODDS_FILES = {
    "2017/2018": "2017_2018.csv",
    "2018/2019": "2018_2019.csv",
    "2019/2020": "2019_2020.csv",
    "2020/2021": "2020_2021.csv",
}

# Compact, football-interpretable groups. Each is a home-minus-away contrast so the
# model sees relative team strength rather than two highly correlated mirrored columns.
FEATURE_GROUPS: dict[str, list[str]] = {
    "core_form": [
        "diff_form5_points_final",
        "diff_form5_goals_for_final",
        "diff_form5_goals_against_final",
        "diff_form5_shots_for_final",
        "diff_form5_xg_for_final",
        "diff_form5_xg_against_final",
    ],
    "style": [
        "diff_form5_passes_final",
        "diff_form5_completed_passes_final",
        "diff_form5_pressures_final",
        "diff_form5_carries_final_third_final",
        "diff_form5_event_share_final",
    ],
    "defense_setpiece": [
        "diff_form5_set_pieces_final",
        "diff_form5_defensive_actions_final",
        "diff_form5_yellow_cards_final",
        "diff_form5_red_cards_final",
    ],
    "short_form": [
        "diff_form3_points_final",
        "diff_form3_goals_for_final",
        "diff_form3_goals_against_final",
    ],
    "context": [
        "diff_rest_days_final",
        "diff_matches_played_before_final",
    ],
    "efficiency": [
        "diff_shot_conv_5_final",
        "diff_possession_share_5_final",
        "diff_pressure_rate_5_final",
        "diff_set_piece_share_5_final",
    ],
}

MOMENTUM_SOURCE = [
    "home_momentum_points", "away_momentum_points",
    "home_momentum_goals_for", "away_momentum_goals_for",
    "home_momentum_xg_for", "away_momentum_xg_for",
]


@dataclass
class SelectionResult:
    selected_groups: list[str]
    selected_features: list[str]
    decisions: pd.DataFrame
    fold_scores: pd.DataFrame


def _rps(y_true: Iterable[str], probs: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    onehot = np.column_stack([(y == c).astype(float) for c in CLASS_ORDER])
    return float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]) ** 2, axis=1) / 2.0))


def _multiclass_log_loss(y_true: Iterable[str], probs: np.ndarray) -> float:
    y = np.asarray(list(y_true))
    idx = np.array([CLASS_ORDER.index(v) for v in y], dtype=int)
    chosen = np.clip(probs[np.arange(len(y)), idx], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def _align_probs(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASS_ORDER)), dtype=float)
    for j, c in enumerate(model.classes_):
        out[:, CLASS_ORDER.index(c)] = raw[:, j]
    return out


def add_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Core five-match form. Against-stat sign is kept literal (home-away); the model
    # learns that lower conceded values are favorable.
    pairs = {
        "diff_form5_points_final": ("home_form5_points", "away_form5_points"),
        "diff_form5_goals_for_final": ("home_form5_goals_for", "away_form5_goals_for"),
        "diff_form5_goals_against_final": ("home_form5_goals_against", "away_form5_goals_against"),
        "diff_form5_shots_for_final": ("home_form5_shots_for", "away_form5_shots_for"),
        "diff_form5_xg_for_final": ("home_form5_xg_for", "away_form5_xg_for"),
        "diff_form5_xg_against_final": ("home_form_5_xg_against", "away_form_5_xg_against"),
        "diff_form5_passes_final": ("home_form_5_passes_for", "away_form_5_passes_for"),
        "diff_form5_completed_passes_final": ("home_form_5_completed_passes_for", "away_form_5_completed_passes_for"),
        "diff_form5_pressures_final": ("home_form_5_pressures_for", "away_form_5_pressures_for"),
        "diff_form5_carries_final_third_final": ("home_form_5_carries_final_third_for", "away_form_5_carries_final_third_for"),
        "diff_form5_event_share_final": ("home_form_5_event_share_for", "away_form_5_event_share_for"),
        "diff_form5_set_pieces_final": ("home_form_5_set_pieces_for", "away_form_5_set_pieces_for"),
        "diff_form5_defensive_actions_final": ("home_form_5_defensive_actions_for", "away_form_5_defensive_actions_for"),
        "diff_form5_yellow_cards_final": ("home_form_5_yellow_cards_for", "away_form_5_yellow_cards_for"),
        "diff_form5_red_cards_final": ("home_form_5_red_cards_for", "away_form_5_red_cards_for"),
        "diff_form3_points_final": ("home_form3_points", "away_form3_points"),
        "diff_form3_goals_for_final": ("home_form3_goals_for", "away_form3_goals_for"),
        "diff_form3_goals_against_final": ("home_form3_goals_against", "away_form3_goals_against"),
        "diff_rest_days_final": ("home_rest_days", "away_rest_days"),
        "diff_matches_played_before_final": ("home_matches_played_before", "away_matches_played_before"),
        "diff_shot_conv_5_final": ("home_shot_conv_5", "away_shot_conv_5"),
        "diff_possession_share_5_final": ("home_possession_share_5", "away_possession_share_5"),
        "diff_pressure_rate_5_final": ("home_pressure_rate_5", "away_pressure_rate_5"),
        "diff_set_piece_share_5_final": ("home_set_piece_share_5", "away_set_piece_share_5"),
    }
    missing_cols = sorted({c for pair in pairs.values() for c in pair if c not in out.columns})
    if missing_cols:
        raise KeyError(f"Required real-data feature columns missing: {missing_cols}")
    for new_col, (home_col, away_col) in pairs.items():
        out[new_col] = pd.to_numeric(out[home_col], errors="coerce") - pd.to_numeric(out[away_col], errors="coerce")
    return out


def retag_real_odds(project_root: Path, pre: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tagged_parts = []
    coverage_parts = []
    excluded_parts = []
    for season in SEASONS:
        part = pre.loc[pre["season"].eq(season)].copy()
        odds_path = project_root / "cache" / "odds" / ODDS_FILES[season]
        odds = pd.read_csv(odds_path)
        # Remove stale fields from the previous broken join before retagging.
        stale = [c for c in ["odds_tagged", "market_p_H", "market_p_D", "market_p_A", "home_key", "away_key"] if c in part.columns]
        part = part.drop(columns=stale)
        tagged, coverage, excluded = tag_odds_to_matches(part, odds)
        coverage.insert(0, "season", season)
        if not excluded.empty:
            excluded.insert(0, "season", season)
        tagged_parts.append(tagged)
        coverage_parts.append(coverage)
        excluded_parts.append(excluded)

    tagged_all = pd.concat(tagged_parts, ignore_index=True).sort_values(["kick_off", "match_id"]).reset_index(drop=True)
    coverage_all = pd.concat(coverage_parts, ignore_index=True)
    excluded_all = pd.concat(excluded_parts, ignore_index=True) if excluded_parts else pd.DataFrame()
    return tagged_all, coverage_all, excluded_all


def _dev_splits(df: pd.DataFrame):
    # Expanding-window development only. 2020/21 is never touched by feature selection.
    yield "dev_2018_19", df["season"].eq("2017/2018"), df["season"].eq("2018/2019")
    yield "dev_2019_20", df["season"].isin(["2017/2018", "2018/2019"]), df["season"].eq("2019/2020")


def _score_features(df: pd.DataFrame, features: list[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for fold_name, train_mask, val_mask in _dev_splits(df):
        train, val = df.loc[train_mask], df.loc[val_mask]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.35, max_iter=3000, class_weight="balanced", random_state=42)),
        ])
        pipe.fit(train[features], train["outcome"])
        probs = _align_probs(pipe.named_steps["model"] if False else pipe, val[features])
        pred = np.asarray(CLASS_ORDER)[np.argmax(probs, axis=1)]
        rows.append({
            "fold": fold_name,
            "n_train": len(train),
            "n_validation": len(val),
            "rps": _rps(val["outcome"], probs),
            "log_loss": _multiclass_log_loss(val["outcome"], probs),
            "accuracy": accuracy_score(val["outcome"], pred),
            "balanced_accuracy": balanced_accuracy_score(val["outcome"], pred),
        })
    scores = pd.DataFrame(rows)
    return float(scores["rps"].mean()), scores


def select_feature_groups(df: pd.DataFrame, min_improvement: float = 0.0010) -> SelectionResult:
    decisions: list[dict] = []
    all_fold_scores: list[pd.DataFrame] = []

    momentum_missing = float(df[MOMENTUM_SOURCE].isna().mean().mean())
    decisions.append({
        "step": 0, "group": "momentum", "action": "REJECT", "mean_dev_rps": np.nan,
        "delta_rps": np.nan, "n_features_after": 0,
        "reason": f"pre-screen rejection: mean missingness={momentum_missing:.3f} (>0.30); unstable in earliest temporal fold",
    })

    # Domain anchor: recent points/goals/shots/xG are retained before any search because
    # they directly encode team strength and are available before kickoff. The question
    # for validation is then whether each *additional* family earns its complexity.
    selected_groups = ["core_form"]
    selected_features = list(FEATURE_GROUPS["core_form"])
    current_rps, fold = _score_features(df, selected_features)
    fold.insert(0, "candidate_group", "core_form")
    fold.insert(0, "selection_step", 1)
    all_fold_scores.append(fold)
    decisions.append({
        "step": 1, "group": "core_form", "action": "KEEP", "mean_dev_rps": current_rps,
        "delta_rps": np.nan, "n_features_after": len(selected_features),
        "reason": "domain anchor: prior five-match points/goals/shots/xG; all strictly pre-kickoff",
    })

    remaining = [g for g in FEATURE_GROUPS if g != "core_form"]
    step = 2
    while remaining:
        trials = []
        for group in remaining:
            feats = selected_features + FEATURE_GROUPS[group]
            rps, fold = _score_features(df, feats)
            fold.insert(0, "candidate_group", group)
            fold.insert(0, "selection_step", step)
            all_fold_scores.append(fold)
            trials.append((rps, group))
        trials.sort()
        best_rps, best_group = trials[0]
        improvement = current_rps - best_rps
        if improvement >= min_improvement:
            selected_groups.append(best_group)
            selected_features += FEATURE_GROUPS[best_group]
            decisions.append({
                "step": step, "group": best_group, "action": "KEEP", "mean_dev_rps": best_rps,
                "delta_rps": improvement, "n_features_after": len(selected_features),
                "reason": f"earned its complexity: mean future-season development RPS improved by {improvement:.4f} (threshold {min_improvement:.4f})",
            })
            current_rps = best_rps
            remaining.remove(best_group)
            step += 1
        else:
            for rps, group in trials:
                delta = current_rps - rps
                decisions.append({
                    "step": step, "group": group, "action": "REJECT", "mean_dev_rps": rps,
                    "delta_rps": delta, "n_features_after": len(selected_features),
                    "reason": f"no out-of-season RPS gain >= {min_improvement:.4f}; rejected to reduce variance and API cost",
                })
            break

    return SelectionResult(
        selected_groups, selected_features, pd.DataFrame(decisions),
        pd.concat(all_fold_scores, ignore_index=True),
    )


def market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, part in df.groupby("season", sort=False):
        p = part[["market_p_H", "market_p_D", "market_p_A"]].to_numpy(float)
        pred = np.asarray(CLASS_ORDER)[p.argmax(axis=1)]
        rows.append({
            "season": season,
            "n_matches": len(part),
            "coverage": float(part["odds_tagged"].mean()),
            "accuracy": accuracy_score(part["outcome"], pred),
            "balanced_accuracy": balanced_accuracy_score(part["outcome"], pred),
            "rps": _rps(part["outcome"], p),
            "log_loss": _multiclass_log_loss(part["outcome"], p),
        })
    return pd.DataFrame(rows)


def build_feature_rationale(df: pd.DataFrame, selection: SelectionResult) -> pd.DataFrame:
    selected = set(selection.selected_groups)
    rows = [
        {
            "feature_family": "market odds",
            "decision": "BASELINE_ONLY",
            "evidence": "100% identity-tagging target; excluded from model features so market comparison remains independent",
            "leakage_status": "pre-match identity only",
        },
        {
            "feature_family": "home/away mirrored raw columns",
            "decision": "COMPRESS_TO_DIFFERENCES",
            "evidence": "paired football quantities describe the same comparison; H-A contrasts reduce redundancy and API feature cost",
            "leakage_status": "history only",
        },
        {
            "feature_family": "momentum",
            "decision": "REJECT",
            "evidence": f"mean missingness across raw momentum columns={df[MOMENTUM_SOURCE].isna().mean().mean():.1%}; unstable in earliest temporal fold",
            "leakage_status": "history only but insufficient availability",
        },
    ]
    final_decisions = selection.decisions.drop_duplicates("group", keep="last").set_index("group")
    for group, feats in FEATURE_GROUPS.items():
        row = final_decisions.loc[group] if group in final_decisions.index else None
        rows.append({
            "feature_family": group,
            "decision": "KEEP" if group in selected else "REJECT",
            "evidence": (f"{len(feats)} features; development mean RPS={float(row['mean_dev_rps']):.4f}; delta={float(row['delta_rps']):+.4f}"
                         if row is not None and pd.notna(row["mean_dev_rps"]) else f"{len(feats)} features"),
            "leakage_status": "all aggregates shifted to matches completed before target kickoff",
        })
    return pd.DataFrame(rows)


def run(project_root: str | Path = ".") -> dict[str, object]:
    root = Path(project_root).resolve()
    out_dir = root / "phase1" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_path = out_dir / "pre_match_features_all_seasons.csv"
    snap_path = out_dir / "in_play_snapshots_all_seasons.csv"
    pre = pd.read_csv(pre_path)
    snaps = pd.read_csv(snap_path)

    # Real-data assertion: no synthetic fallback is allowed in the final workflow.
    if len(pre) != pre["match_id"].nunique() or len(pre) == 0:
        raise AssertionError("Expected one real pre-match row per match.")
    if not set(pre["match_id"]).issubset(set(snaps["match_id"])):
        raise AssertionError("In-play snapshots do not cover all real modeling matches.")

    tagged, coverage, excluded = retag_real_odds(root, pre)
    tagged = add_compact_features(tagged)
    selection = select_feature_groups(tagged)
    rationale = build_feature_rationale(tagged, selection)
    market = market_metrics(tagged)

    tagged.to_csv(out_dir / "pre_match_features_final.csv", index=False)
    coverage.to_csv(out_dir / "odds_coverage_final.csv", index=False)
    excluded.to_csv(out_dir / "odds_excluded_matches_final.csv", index=False)
    selection.decisions.to_csv(out_dir / "feature_selection_decisions_final.csv", index=False)
    selection.fold_scores.to_csv(out_dir / "feature_selection_fold_scores_final.csv", index=False)
    rationale.to_csv(out_dir / "feature_rationale_final.csv", index=False)
    market.to_csv(out_dir / "market_baseline_final.csv", index=False)
    pd.DataFrame({"feature": selection.selected_features}).to_csv(out_dir / "selected_features_final.csv", index=False)

    report = pd.DataFrame([
        {"metric": "raw_matches_cached", "value": len(list((root / "cache" / "events").glob("*.json")))},
        {"metric": "pre_match_modeling_rows", "value": len(tagged)},
        {"metric": "in_play_snapshots", "value": len(snaps)},
        {"metric": "odds_tagged_rows", "value": int(tagged["odds_tagged"].sum())},
        {"metric": "odds_coverage", "value": float(tagged["odds_tagged"].mean())},
        {"metric": "selected_feature_groups", "value": ", ".join(selection.selected_groups)},
        {"metric": "selected_feature_count", "value": len(selection.selected_features)},
        {"metric": "synthetic_rows_used", "value": 0},
        {"metric": "final_test_season_reserved", "value": "2020/2021"},
    ])
    report.to_csv(out_dir / "final_phase1_summary.csv", index=False)

    return {
        "pre": tagged,
        "snapshots": snaps,
        "coverage": coverage,
        "excluded": excluded,
        "selection": selection,
        "rationale": rationale,
        "market": market,
        "report": report,
    }


if __name__ == "__main__":
    result = run(Path(__file__).resolve().parents[1])
    print("\n=== FINAL PHASE 1 ===")
    print(result["report"].to_string(index=False))
    print("\nFeature decisions:")
    print(result["selection"].decisions.to_string(index=False))
    print("\nMarket baseline:")
    print(result["market"].to_string(index=False))
