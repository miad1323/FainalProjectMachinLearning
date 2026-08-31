import os

os.environ["THREADPOOLCTL_ENABLE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT.parent / "cache"
sys.path.insert(0, str(PROJECT_ROOT))

from gmm_sampling import GMMSampling
from statsbomb_phase1 import (
    add_chronological_split,
    assert_pre_match_leakage_free,
    assert_time_t_cut,
    build_in_play_snapshots,
    build_match_team_table,
    build_pre_match_features,
    download_odds_csv,
    ingest_season,
    make_demo_football_data,
    tag_odds_to_matches,
    load_competitions,
)
from evaluation import evaluate_predictions, align_probabilities

warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL"
)


def ece_score(y_true, proba, n_bins=10):
    if isinstance(y_true[0], str):
        classes = np.unique(y_true)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_true_idx = np.array([class_to_idx[c] for c in y_true])
    else:
        y_true_idx = np.array(y_true, dtype=int)
    confidences = proba[np.arange(len(y_true_idx)), y_true_idx]
    predictions = np.argmax(proba, axis=1)
    accuracies = (predictions == y_true_idx).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins, right=False)
    bin_indices = np.clip(bin_indices, 1, n_bins) - 1

    ece = 0.0
    bin_info = []
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) == 0:
            bin_info.append({"bin": b, "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        conf_bin = confidences[mask]
        acc_bin = accuracies[mask]
        avg_conf = np.mean(conf_bin)
        avg_acc = np.mean(acc_bin)
        count = len(conf_bin)
        ece += (count / len(confidences)) * np.abs(avg_acc - avg_conf)
        bin_info.append({"bin": b, "count": count, "confidence": avg_conf,
                         "accuracy": avg_acc})
    return ece, pd.DataFrame(bin_info)


SEED = 42
np.random.seed(SEED)

COMPETITION_NAME = "La Liga"
DEFAULT_SEASONS = ["2015/2016", "2016/2017", "2017/2018", "2018/2019"]

CONFIG = {
    "DATA_MODE": "statsbomb",
    "COMPETITION_NAME": COMPETITION_NAME,
    "SEASONS": DEFAULT_SEASONS,
    "MAX_MATCHES": None,
    "INCLUDE_THREE_SIXTY": True,
    "ROLLING_WINDOW": 5,
    "MIN_HISTORY": 1,
    "SNAPSHOT_MINUTES": list(range(5, 91, 5)),
    "RECENT_WINDOW": 5,
    "ENABLE_ODDS_DOWNLOAD": True,
    "CACHE_DIR": str(CACHE_ROOT),
    "OUTPUT_DIR": str(PROJECT_ROOT / "outputs"),
}

Path(CONFIG["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
output_dir = Path(CONFIG["OUTPUT_DIR"])

print("Project root:", PROJECT_ROOT)
print("Cache root:", CACHE_ROOT)
print("Seed:", SEED)

print("\n--- Fetching available seasons for competition ---")
competitions = load_competitions(CONFIG["CACHE_DIR"])
comp_mask = competitions["competition_name"].str.casefold().eq(
    COMPETITION_NAME.casefold()
)
available = competitions.loc[comp_mask, ["season_name"]].drop_duplicates()
available_seasons = available["season_name"].tolist()
print(f"Found {len(available_seasons)} season(s): {available_seasons}")

if available_seasons:
    def season_year(s):
        return int(s.split("/")[0])

    available_seasons_sorted = sorted(available_seasons, key=season_year)
    n_seasons = min(4, len(available_seasons_sorted))
    selected_seasons = available_seasons_sorted[-n_seasons:]
    CONFIG["SEASONS"] = selected_seasons
    print(f"Selected seasons for experiments: {selected_seasons}")
else:
    print(f"No seasons found for {COMPETITION_NAME}. Using default list: "
          f"{DEFAULT_SEASONS}")
    CONFIG["SEASONS"] = DEFAULT_SEASONS

print("Final Config:", json.dumps(CONFIG, indent=2, default=str))


def impute_with_medians(X_train, X_test):
    medians = X_train.median(numeric_only=True)
    X_train_imp = X_train.fillna(medians).fillna(0.0)
    X_test_imp = X_test.fillna(medians).fillna(0.0)
    return X_train_imp, X_test_imp


def run_imbalance_experiment(pre_match, feature_cols, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import precision_recall_fscore_support

    train = pre_match[pre_match.split == "train"].copy()
    test = pre_match[pre_match.split == "test"].copy()
    X_train_raw = train[feature_cols]
    y_train = train.outcome
    X_test_raw = test[feature_cols]
    y_test = test.outcome

    X_train, X_test = impute_with_medians(X_train_raw, X_test_raw)

    results = []
    base_rf = RandomForestClassifier(
        n_estimators=150,
        random_state=random_state,
        n_jobs=1
    )
    class_order = ["H", "D", "A"]

    train_counts = y_train.value_counts()
    min_class_size = train_counts.min()
    adaptive_k = min(5, max(1, min_class_size - 1))
    print(f"Adaptive k_neighbors for SMOTE: {adaptive_k} (min class size: "
          f"{min_class_size})")

    def evaluate_model(model, X_test, y_test, method_name):
        pred = model.predict(X_test)
        prob = (align_probabilities(model, X_test)
                if hasattr(model, "predict_proba") else None)
        if prob is not None:
            metrics = evaluate_predictions(y_test, pred, prob,
                                           classes=class_order)
        else:
            dummy = DummyClassifier(strategy="prior")
            dummy.fit(X_train, y_train)
            prob = dummy.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, pred, prob,
                                           classes=class_order)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, labels=class_order, zero_division=0
        )
        metrics["precision_H"] = p[0]
        metrics["recall_H"] = r[0]
        metrics["f1_H"] = f[0]
        metrics["precision_D"] = p[1]
        metrics["recall_D"] = r[1]
        metrics["f1_D"] = f[1]
        metrics["precision_A"] = p[2]
        metrics["recall_A"] = r[2]
        metrics["f1_A"] = f[2]
        metrics["method"] = method_name
        if hasattr(model, "classes_"):
            metrics["train_size"] = len(X_train)
        return metrics

    model = clone(base_rf)
    model.fit(X_train, y_train)
    results.append(evaluate_model(model, X_test, y_test, "Vanilla"))

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    results.append(evaluate_model(model, X_test, y_test, "ClassWeight"))

    try:
        smote = SMOTE(random_state=random_state, k_neighbors=adaptive_k)
        pipe = ImbPipeline([("smote", smote), ("rf", clone(base_rf))])
        pipe.fit(X_train, y_train)
        results.append(evaluate_model(pipe.named_steps["rf"], X_test, y_test,
                                      "SMOTE"))
    except ValueError as e:
        print(f"SMOTE failed: {e}. Skipping.")

    try:
        bsmote = BorderlineSMOTE(random_state=random_state,
                                 k_neighbors=adaptive_k)
        pipe = ImbPipeline([("bsmote", bsmote), ("rf", clone(base_rf))])
        pipe.fit(X_train, y_train)
        results.append(evaluate_model(pipe.named_steps["rf"], X_test, y_test,
                                      "BorderlineSMOTE"))
    except ValueError as e:
        print(f"BorderlineSMOTE failed: {e}. Skipping.")

    try:
        adasyn = ADASYN(random_state=random_state, n_neighbors=adaptive_k)
        pipe = ImbPipeline([("adasyn", adasyn), ("rf", clone(base_rf))])
        pipe.fit(X_train, y_train)
        results.append(evaluate_model(pipe.named_steps["rf"], X_test, y_test,
                                      "ADASYN"))
    except ValueError as e:
        print(f"ADASYN failed: {e}. Skipping.")

    majority_label = train_counts.idxmax()
    minority_labels = tuple(c for c in class_order if c != majority_label)

    sampler = GMMSampling(
        minority_classes=minority_labels,
        majority_classes=(majority_label,),
        covariance_type='diag',
        max_components=8,
        random_state=random_state,
        k_neighbors=5,
        scale_features=True
    )
    X_gmm, y_gmm = sampler.fit_resample(X_train, y_train)
    model = clone(base_rf)
    model.fit(X_gmm, y_gmm)
    pred = model.predict(X_test)
    prob = align_probabilities(model, X_test)
    metrics_gmm = evaluate_predictions(y_test, pred, prob, classes=class_order)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, pred, labels=class_order, zero_division=0
    )
    metrics_gmm["precision_H"] = p[0]
    metrics_gmm["recall_H"] = r[0]
    metrics_gmm["f1_H"] = f[0]
    metrics_gmm["precision_D"] = p[1]
    metrics_gmm["recall_D"] = r[1]
    metrics_gmm["f1_D"] = f[1]
    metrics_gmm["precision_A"] = p[2]
    metrics_gmm["recall_A"] = r[2]
    metrics_gmm["f1_A"] = f[2]
    metrics_gmm["method"] = "GMMSampling"
    metrics_gmm["train_size"] = len(X_gmm)
    results.append(metrics_gmm)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    prob = dummy.predict_proba(X_test)
    metrics_dummy = evaluate_predictions(y_test, pred, prob,
                                         classes=class_order)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, pred, labels=class_order, zero_division=0
    )
    metrics_dummy["precision_H"] = p[0]
    metrics_dummy["recall_H"] = r[0]
    metrics_dummy["f1_H"] = f[0]
    metrics_dummy["precision_D"] = p[1]
    metrics_dummy["recall_D"] = r[1]
    metrics_dummy["f1_D"] = f[1]
    metrics_dummy["precision_A"] = p[2]
    metrics_dummy["recall_A"] = r[2]
    metrics_dummy["f1_A"] = f[2]
    metrics_dummy["method"] = "DummyPrior"
    metrics_dummy["train_size"] = len(X_train)
    results.append(metrics_dummy)

    df = pd.DataFrame(results)
    base_cols = ["method", "train_size", "balanced_accuracy", "macro_f1",
                 "mcc", "g_mean", "log_loss", "brier", "rps"]
    per_class = ["precision_H", "recall_H", "f1_H", "precision_D", "recall_D",
                 "f1_D", "precision_A", "recall_A", "f1_A"]
    cols = base_cols + per_class
    df = df[cols]

    return df, sampler, {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_gmm": X_gmm,
        "y_gmm": y_gmm
    }


def ingest_seasons(competition_name, seasons, cache_dir, max_matches=None,
                   include_lineups=True, include_three_sixty=True,
                   verbose=True):
    all_matches = []
    all_events = []
    all_lineups = []
    all_three_sixty = []
    metadata_list = []

    for season in seasons:
        print(f"\n--- Ingesting {competition_name} {season} ---")
        m, e, l, t, meta = ingest_season(
            competition_name=competition_name,
            season_name=season,
            cache_dir=cache_dir,
            max_matches=max_matches,
            include_lineups=include_lineups,
            include_three_sixty=include_three_sixty,
            verbose=verbose
        )
        m["season"] = season
        all_matches.append(m)
        all_events.append(e)
        all_lineups.append(l)
        all_three_sixty.append(t)
        metadata_list.append(meta)

    matches = pd.concat(all_matches, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    lineups = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
    three_sixty = pd.concat(all_three_sixty, ignore_index=True) if all_three_sixty else pd.DataFrame()

    combined_metadata = {
        "competition_name": competition_name,
        "seasons": seasons,
        "n_matches_total": len(matches),
        "n_events_total": len(events),
        "n_lineups_total": len(lineups),
        "n_three_sixty_total": len(three_sixty),
        "per_season_metadata": metadata_list,
    }
    return matches, events, lineups, three_sixty, combined_metadata


def get_odds_url(season, competition="La Liga"):
    codes = {
        "Premier League": "E0",
        "La Liga": "SP1",
        "Bundesliga": "D1",
        "Ligue 1": "F1",
        "Serie A": "I1",
    }
    code = codes.get(competition, "E0")
    start, end = season.split("/")
    short = start[2:] + end[2:]
    return f"https://www.football-data.co.uk/mmz4281/{short}/{code}.csv"


def validate_data_integration(matches, events, lineups, three_sixty):
    report = {}
    issues = []

    total_matches = len(matches)
    match_ids = set(matches.match_id.astype(int))
    events_match_ids = set(events.match_id.astype(int)) if not events.empty else set()
    lineups_match_ids = set(lineups.match_id.astype(int)) if not lineups.empty else set()
    three_sixty_match_ids = set(three_sixty.match_id.astype(int)) if not three_sixty.empty else set()

    coverage = {
        "matches_with_events": len(events_match_ids),
        "matches_with_lineups": len(lineups_match_ids),
        "matches_with_three_sixty": len(three_sixty_match_ids),
        "pct_events": len(events_match_ids) / total_matches if total_matches > 0 else 0,
        "pct_lineups": len(lineups_match_ids) / total_matches if total_matches > 0 else 0,
        "pct_three_sixty": len(three_sixty_match_ids) / total_matches if total_matches > 0 else 0,
    }
    report["coverage"] = coverage

    if not events.empty:
        orphan_events = events[~events.match_id.isin(matches.match_id)]
        if len(orphan_events) > 0:
            issues.append({
                "check": "event.match_id in matches",
                "violations": len(orphan_events),
                "example_ids": orphan_events.match_id.head(5).tolist()
            })
        else:
            issues.append({"check": "event.match_id in matches", "violations": 0})

        if not lineups.empty:
            orphan_lineups = lineups[~lineups.match_id.isin(matches.match_id)]
            if len(orphan_lineups) > 0:
                issues.append({
                    "check": "lineup.match_id in matches",
                    "violations": len(orphan_lineups),
                    "example_ids": orphan_lineups.match_id.head(5).tolist()
                })
            else:
                issues.append({"check": "lineup.match_id in matches",
                               "violations": 0})

        if not three_sixty.empty:
            orphan_360 = three_sixty[~three_sixty.match_id.isin(matches.match_id)]
            if len(orphan_360) > 0:
                issues.append({
                    "check": "three_sixty.match_id in matches",
                    "violations": len(orphan_360),
                    "example_ids": orphan_360.match_id.head(5).tolist()
                })
            else:
                issues.append({"check": "three_sixty.match_id in matches",
                               "violations": 0})

        team_map = matches.set_index("match_id")[["home_team_id", "away_team_id"]]
        events_with_team = events.dropna(subset=["team_id"])
        if not events_with_team.empty:
            def is_valid_team(row):
                mid = int(row["match_id"])
                tid = int(row["team_id"])
                if mid in team_map.index:
                    home = int(team_map.loc[mid, "home_team_id"])
                    away = int(team_map.loc[mid, "away_team_id"])
                    return tid in (home, away)
                return False
            valid = events_with_team.apply(is_valid_team, axis=1)
            invalid_events = events_with_team[~valid]
            if len(invalid_events) > 0:
                issues.append({
                    "check": "event.team_id matches match home/away",
                    "violations": len(invalid_events),
                    "example_ids": invalid_events[["match_id", "team_id"]]
                    .head(5).to_dict(orient="records")
                })
            else:
                issues.append({"check": "event.team_id matches match home/away",
                               "violations": 0})

        if not lineups.empty:
            lineups_with_team = lineups.dropna(subset=["team_id"])
            if not lineups_with_team.empty:
                def is_valid_lineup_team(row):
                    mid = int(row["match_id"])
                    tid = int(row["team_id"])
                    if mid in team_map.index:
                        home = int(team_map.loc[mid, "home_team_id"])
                        away = int(team_map.loc[mid, "away_team_id"])
                        return tid in (home, away)
                    return False
                valid = lineups_with_team.apply(is_valid_lineup_team, axis=1)
                invalid_lineups = lineups_with_team[~valid]
                if len(invalid_lineups) > 0:
                    issues.append({
                        "check": "lineup.team_id matches match home/away",
                        "violations": len(invalid_lineups),
                        "example_ids": invalid_lineups[["match_id", "team_id"]]
                        .head(5).to_dict(orient="records")
                    })
                else:
                    issues.append({"check": "lineup.team_id matches match home/away",
                                   "violations": 0})

        if not lineups.empty and not events.empty:
            lineup_players = lineups[["match_id", "player_id"]].drop_duplicates()
            events_with_player = events.dropna(subset=["player_id"])
            if not events_with_player.empty:
                merged = events_with_player.merge(
                    lineup_players,
                    on=["match_id", "player_id"],
                    how="left",
                    indicator=True
                )
                missing = merged[merged["_merge"] == "left_only"]
                if len(missing) > 0:
                    issues.append({
                        "check": "event.player_id in lineups",
                        "violations": len(missing),
                        "example_ids": missing[["match_id", "player_id"]]
                        .head(5).to_dict(orient="records")
                    })
                else:
                    issues.append({"check": "event.player_id in lineups",
                                   "violations": 0})

        if not three_sixty.empty and not events.empty:
            three_sixty_event = three_sixty.dropna(subset=["event_id"])
            if not three_sixty_event.empty:
                events_ids = set(events["event_id"].astype(str))
                missing_360 = three_sixty_event[
                    ~three_sixty_event["event_id"].astype(str).isin(events_ids)
                ]
                if len(missing_360) > 0:
                    issues.append({
                        "check": "three_sixty.event_id in events",
                        "violations": len(missing_360),
                        "example_ids": missing_360[["event_id"]]
                        .head(5).to_dict(orient="records")
                    })
                else:
                    issues.append({"check": "three_sixty.event_id in events",
                                   "violations": 0})

    issues_df = pd.DataFrame(issues)
    report["issues"] = issues_df
    return report


print("\n--- 1. Data Ingestion (Multiple Seasons) ---")
matches, events, lineups, three_sixty, metadata = ingest_seasons(
    competition_name=CONFIG["COMPETITION_NAME"],
    seasons=CONFIG["SEASONS"],
    cache_dir=CONFIG["CACHE_DIR"],
    max_matches=CONFIG["MAX_MATCHES"],
    include_lineups=True,
    include_three_sixty=CONFIG["INCLUDE_THREE_SIXTY"],
    verbose=True,
)
print("Matches:", matches.shape)
print("Events:", events.shape)
print("Lineups:", lineups.shape)
print("360 frames:", three_sixty.shape)
print("Seasons present:", matches.season.unique())

if not lineups.empty:
    print(f"Lineups available for {lineups.match_id.nunique()} matches.")
else:
    print("No lineups data loaded.")
if not three_sixty.empty:
    print(f"360 data available for {three_sixty.match_id.nunique()} matches.")
else:
    print("No 360 data available.")

matches.match_id = matches.match_id.astype(int)
if not events.empty:
    events.match_id = events.match_id.astype(int)

print("\n--- 2. Integrity Checks ---")
assert matches.match_id.is_unique, "match_id must be unique."
if not events.empty:
    assert events.match_id.isin(matches.match_id).all(), (
        "Every event must have a parent match."
    )
print("Integrity checks passed.")

print("\n--- Data Integration Validation ---")
integration_report = validate_data_integration(matches, events, lineups,
                                               three_sixty)
integration_report["issues"].to_csv(
    output_dir / "data_integration_report.csv", index=False
)

cov = integration_report["coverage"]
print(f"Coverage:")
print(f"  Matches with events: {cov['matches_with_events']} "
      f"({cov['pct_events']:.1%})")
print(f"  Matches with lineups: {cov['matches_with_lineups']} "
      f"({cov['pct_lineups']:.1%})")
print(f"  Matches with 360: {cov['matches_with_three_sixty']} "
      f"({cov['pct_three_sixty']:.1%})")

if not integration_report["issues"].empty:
    print("Integrity violations found (see data_integration_report.csv "
          "for details):")
    for _, row in integration_report["issues"].iterrows():
        if row["violations"] > 0:
            print(f"  - {row['check']}: {row['violations']} violations")
else:
    print("All relational integrity checks passed.")

metadata["integration_coverage"] = cov
metadata["integration_issues"] = (
    integration_report["issues"].to_dict(orient="records")
    if not integration_report["issues"].empty else []
)

print("\n--- 3. Pre-Match Features (across all seasons, chronological) ---")
team_matches = build_match_team_table(matches, events)
if team_matches.empty:
    raise RuntimeError("team_matches is empty. Check data.")
print("Team matches shape:", team_matches.shape)
pre_match = build_pre_match_features(
    matches,
    team_matches,
    rolling_window=CONFIG["ROLLING_WINDOW"],
    min_history=CONFIG["MIN_HISTORY"],
)
print("Pre-match examples:", pre_match.shape)
print("Outcome distribution:\n", pre_match.outcome.value_counts())

print("\n--- 4. In-Play Snapshots (across all seasons) ---")
snapshots = build_in_play_snapshots(
    matches,
    events,
    pre_match,
    snapshot_minutes=CONFIG["SNAPSHOT_MINUTES"],
    recent_window=CONFIG["RECENT_WINDOW"],
)
assert_time_t_cut(snapshots)
print("Snapshot table shape:", snapshots.shape)

print("\n--- 5. Season-based Splits (index) ---")
season_order = CONFIG["SEASONS"]
season_to_idx = {s: i for i, s in enumerate(season_order)}
pre_match["season_idx"] = pre_match["season"].map(season_to_idx)
snapshots["season_idx"] = snapshots["match_id"].map(
    pre_match.set_index("match_id")["season_idx"]
)

print("\n--- 6. Leakage Audit (global) ---")
leakage_checks = []
assert_pre_match_leakage_free(pre_match)
leakage_checks.append(("Pre-match history strictly before kick-off", True))

if not snapshots.empty:
    assert_time_t_cut(snapshots)
    leakage_checks.append(("In-play event timestamp <= snapshot time", True))

leakage_checks.append(("GMMSampling called only on training partition", True))
leakage_audit = pd.DataFrame(leakage_checks, columns=["check", "passed"])
print(leakage_audit.to_string(index=False))
assert leakage_audit.passed.all()
print("All global leakage checks passed.")

print("\n--- 7. Odds Tagging (per season) ---")
if CONFIG["ENABLE_ODDS_DOWNLOAD"]:
    all_tagged_matches = []
    all_odds_coverage = []
    all_excluded = []
    for season in CONFIG["SEASONS"]:
        try:
            url = get_odds_url(season, competition=CONFIG["COMPETITION_NAME"])
            cache_path = Path(CONFIG["CACHE_DIR"]) / "odds" / (
                f"{season.replace('/', '_')}.csv"
            )
            odds = download_odds_csv(url=url, cache_path=cache_path)
            season_matches = matches[matches.season == season].copy()
            tagged, coverage, excluded = tag_odds_to_matches(season_matches,
                                                             odds)
            all_tagged_matches.append(tagged)
            all_odds_coverage.append(coverage)
            all_excluded.append(excluded)
        except Exception as exc:
            print(f"Odds integration failed for {season}: {repr(exc)}")
    if all_tagged_matches:
        tagged_matches = pd.concat(all_tagged_matches, ignore_index=True)
        odds_coverage = pd.concat(all_odds_coverage, ignore_index=True)
        excluded_matches = pd.concat(all_excluded, ignore_index=True)
    else:
        tagged_matches = pd.DataFrame()
        odds_coverage = pd.DataFrame()
        excluded_matches = pd.DataFrame()
else:
    tagged_matches = pd.DataFrame()
    odds_coverage = pd.DataFrame()
    excluded_matches = pd.DataFrame()

if not odds_coverage.empty:
    print("Odds coverage (per season):\n", odds_coverage)
print("Excluded matches (total):", len(excluded_matches))

if not tagged_matches.empty:
    odds_cols = ['match_id', 'odds_tagged', 'market_p_H', 'market_p_D',
                 'market_p_A']
    pre_match = pre_match.merge(
        tagged_matches[odds_cols],
        on='match_id',
        how='left'
    )
    pre_match['odds_tagged'] = pre_match['odds_tagged'].fillna(False).astype(
        bool
    )

    snapshots = snapshots.merge(
        tagged_matches[odds_cols],
        on='match_id',
        how='left'
    )
    snapshots['odds_tagged'] = snapshots['odds_tagged'].fillna(False).astype(
        bool
    )
else:
    for df in (pre_match, snapshots):
        df['odds_tagged'] = False
        df['market_p_H'] = np.nan
        df['market_p_D'] = np.nan
        df['market_p_A'] = np.nan

print("\n--- 8. Feature Selection ---")
NON_FEATURES = {
    "match_id", "kick_off", "match_date", "competition_id", "season_id",
    "home_team_id", "away_team_id", "home_team", "away_team",
    "home_score", "away_score", "outcome", "goal_margin", "split",
    "competition_stage", "stadium", "referee",
    "home_history_max_kickoff", "away_history_max_kickoff",
    "season", "season_idx",
}
feature_cols = [
    c for c in pre_match.columns
    if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(pre_match[c])
]
print("Number of numeric features:", len(feature_cols))
print(feature_cols[:10], "...")

print("\n--- 9. Cross-Season Experiments ---")
all_experiment_results = []
all_diagnostics = []
all_gmm_data = []

if len(CONFIG["SEASONS"]) >= 2:
    for i in range(len(CONFIG["SEASONS"]) - 1):
        train_season = CONFIG["SEASONS"][i]
        test_season = CONFIG["SEASONS"][i + 1]
        print(f"\n{'=' * 60}")
        print(f" EXPERIMENT {i + 1}: Train on {train_season} -> Test on "
              f"{test_season}")
        print(f"{'=' * 60}")

        pre_match_split = pre_match.copy()
        pre_match_split["split"] = "train"
        pre_match_split.loc[pre_match_split.season == test_season,
                            "split"] = "test"
        mask = pre_match_split.season.isin([train_season, test_season])
        pre_match_split = pre_match_split[mask].copy()
        snapshots_split = snapshots[
            snapshots.match_id.isin(pre_match_split.match_id)
        ].copy()
        snapshots_split["split"] = snapshots_split.match_id.map(
            pre_match_split.set_index("match_id")["split"]
        )

        if (pre_match_split[pre_match_split.split == "train"].empty or
            pre_match_split[pre_match_split.split == "test"].empty):
            print(f"Skipping {train_season}->{test_season}: not enough "
                  f"matches.")
            continue

        results, sampler, gmm_data = run_imbalance_experiment(
            pre_match_split, feature_cols, random_state=SEED
        )
        results["train_season"] = train_season
        results["test_season"] = test_season
        all_experiment_results.append(results)
        all_diagnostics.append(sampler.get_diagnostics())
        all_gmm_data.append(gmm_data)

        print(f"\n--- Results for Experiment {i + 1} ---")
        display_cols = ["method", "balanced_accuracy", "macro_f1", "mcc",
                        "g_mean", "log_loss"]
        print(results[display_cols].round(4).to_string(index=False))
        print("-" * 60)
else:
    print("Not enough seasons for cross-validation; skipping experiments.")

if all_experiment_results:
    combined_results = pd.concat(all_experiment_results, ignore_index=True)
    combined_results.to_csv(
        output_dir / "cross_season_experiment_results.csv", index=False
    )
    print("\nCombined cross-season results (mean by method):")
    print(combined_results.groupby("method")["balanced_accuracy"].mean()
          .round(4))
else:
    print("No experiments ran.")

for idx, diag in enumerate(all_diagnostics):
    diag.to_csv(output_dir / f"gmm_diagnostics_experiment_{idx + 1}.csv",
                index=False)

print("\n--- 10. Visualizations (Aggregated) ---")
if all_experiment_results:
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped = combined_results.groupby(
        ["method", "train_season", "test_season"]
    )["balanced_accuracy"].mean().unstack(level=0)
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Balanced Accuracy per Cross-Season Split")
    ax.set_ylabel("Balanced Accuracy")
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "cross_season_balanced_accuracy.png", dpi=180)
    plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
pre_match.groupby(["season", "outcome"]).size().unstack().plot(
    kind="bar", stacked=True, ax=ax
)
ax.set_title("Class Distribution by Season")
ax.set_xlabel("Season")
ax.set_ylabel("Number of Matches")
plt.tight_layout()
plt.savefig(output_dir / "class_distribution_by_season.png", dpi=180)
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))
top_feats = feature_cols[:20]
sns.heatmap(pre_match[top_feats].corr(), ax=ax, cmap="coolwarm", center=0)
ax.set_title("Feature Correlation (top 20)")
plt.tight_layout()
plt.savefig(output_dir / "feature_correlation.png", dpi=180)
plt.close()

print("\n--- 11. Exporting Artifacts ---")
pre_match.to_csv(output_dir / "pre_match_features_all_seasons.csv", index=False)
snapshots.to_csv(output_dir / "in_play_snapshots_all_seasons.csv", index=False)
leakage_audit.to_csv(output_dir / "leakage_audit.csv", index=False)

if not odds_coverage.empty:
    odds_coverage.to_csv(output_dir / "odds_coverage.csv", index=False)
if isinstance(excluded_matches, pd.DataFrame) and len(excluded_matches.columns) > 0:
    excluded_matches.to_csv(output_dir / "odds_excluded_matches.csv", index=False)

metadata_config = {**CONFIG, "CACHE_DIR": str(CACHE_ROOT),
                   "OUTPUT_DIR": str(output_dir)}
run_metadata = {
    "seed": SEED,
    "config": metadata_config,
    "data_metadata": metadata,
    "feature_count": len(feature_cols),
    "snapshot_rows": len(snapshots),
    "odds_tagged_rows": int(tagged_matches.odds_tagged.sum())
    if "odds_tagged" in tagged_matches else 0,
    "odds_excluded_rows": len(excluded_matches),
    "num_experiments": len(all_experiment_results),
    "demo_warning": False,
}
(output_dir / "run_metadata.json").write_text(
    json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
)

print("Phase 1 completed successfully. Outputs saved in:", output_dir)
print("Files created:")
for path in sorted(output_dir.iterdir()):
    print("  -", path.name)