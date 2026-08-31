import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential"
os.environ["THREADPOOLCTL_ENABLE"] = "0"

import matplotlib
matplotlib.use('Agg')
import sys
import gc
import warnings
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2_analysis import generate_analysis
from phase2_experiments import run_all

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The y_prob values do not sum to one")

pre_path = PROJECT_ROOT / 'phase1' / 'outputs' / 'pre_match_features_all_seasons.csv'
snap_path = PROJECT_ROOT / 'phase1' / 'outputs' / 'in_play_snapshots_all_seasons.csv'

if not pre_path.exists():
    raise FileNotFoundError(f"Pre-match file not found: {pre_path}")
if not snap_path.exists():
    raise FileNotFoundError(f"Snapshot file not found: {snap_path}")

pre = pd.read_csv(pre_path)
snap = pd.read_csv(snap_path)

for col in ['outcome', 'goal_margin']:
    if col not in snap.columns:
        snap = snap.merge(pre[['match_id', col]], on='match_id', how='left')
    if snap[col].isnull().any():
        raise ValueError(f"Some snapshots have no matching {col} from pre data.")

seasons = sorted(pre['season'].unique())
print(f"Seasons found: {seasons}")
if len(seasons) < 2:
    raise ValueError("Need at least two seasons for cross-season experiments.")

season_pairs = [(seasons[i], seasons[i + 1]) for i in range(len(seasons) - 1)]
print(f"Will run {len(season_pairs)} splits: {season_pairs}")

base_out = PROJECT_ROOT / 'phase2' / 'outputs'
base_out.mkdir(parents=True, exist_ok=True)

all_results = []
for split_idx, (train_season, test_season) in enumerate(season_pairs, start=1):
    print(f"\n{'=' * 60}")
    print(f" SPLIT {split_idx}: Train on {train_season} -> Test on {test_season}")
    print(f"{'=' * 60}")

    mask = pre['season'].isin([train_season, test_season])
    pre_split = pre.loc[mask].copy()
    match_ids = pre_split['match_id'].unique()
    snap_split = snap.loc[snap['match_id'].isin(match_ids)].copy()

    pre_split['split'] = pre_split['season'].apply(
        lambda s: 'train' if s == train_season else 'test'
    )
    snap_split = snap_split.merge(
        pre_split[['match_id', 'split']], on='match_id', how='left'
    )

    if snap_split['split'].isnull().any():
        raise ValueError("Some snapshots missing split.")

    violations = (
        snap_split['max_event_time_seconds_used'] >
        snap_split['snapshot_time_seconds']
    ).sum()
    if violations:
        print(f"WARNING: {violations} snapshots have time-leakage "
              "(proceeding anyway).")

    out_dir = base_out / f'split_{split_idx}'
    out_dir.mkdir(parents=True, exist_ok=True)

    split_results = run_all(pre_split, snap_split, out_dir / 'phase2')
    gc.collect()

    generate_analysis(pre_split, snap_split, out_dir / 'phase2_analysis')
    gc.collect()

    split_results['train_season'] = train_season
    split_results['test_season'] = test_season
    split_results['split_index'] = split_idx
    all_results.append(split_results)

    print(f"\n--- Results for Split {split_idx} ---")
    display_cols = [
        c for c in ["task", "model", "calibration", "balanced_accuracy",
                    "macro_f1", "mcc", "mae", "rmse", "r2"]
        if c in split_results.columns
    ]
    print(split_results[display_cols].round(4).to_string(index=False))
    print("-" * 60)

    del pre_split, snap_split, split_results
    gc.collect()

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(base_out / 'phase2_cross_season_results.csv', index=False)
    print("\n=== Average balanced accuracy per model (across splits) ===")
    print(combined.groupby(['model', 'task', 'calibration'])['balanced_accuracy']
          .mean().round(4))
else:
    print("No results collected.")