from __future__ import annotations
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings("ignore")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.inspection import permutation_importance
from phase2_experiments import CLASS_ORDER, classification_metrics, ece_score

try:
    import shap
except ImportError:
    print("=" * 70)
    print("ERROR: SHAP is required for this analysis but not installed.")
    print("Please install it manually in your environment:")
    print("  pip install shap")
    print("  or")
    print("  conda install -c conda-forge shap")
    print("Then rerun the script.")
    print("=" * 70)
    raise ImportError("SHAP not installed. Please install it manually and rerun.")
SHAP = shap


def classifier_factory(name, seed=42):
    if name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight='balanced'
        )
    if name == 'GBM':
        return GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed
        )
    if name == 'SVC':
        return SVC(C=1.0, gamma='scale', probability=True, random_state=seed)
    if name == 'XGBoost':
        return __import__('xgboost').XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=seed
        )
    if name == 'LightGBM':
        return __import__('lightgbm').LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            verbose=-1
        )
    raise KeyError(name)


def train_calibrated(name, X, y, seed=42):
    m = classifier_factory(name, seed)
    ycode = pd.Categorical(y, categories=CLASS_ORDER).codes
    cal = CalibratedClassifierCV(m, method='sigmoid', cv=3, ensemble=True)
    cal.fit(X, ycode)
    return cal


def generate_reliability_plots(out: Path):
    pred_files = list(out.glob("predictions_*.csv"))
    for f in pred_files:
        df = pd.read_csv(f)
        if not all(col in df.columns for col in ['p_H', 'p_D', 'p_A', 'y_true']):
            continue
        stem = f.stem
        parts = stem.split('_')
        if len(parts) < 3:
            continue
        task = parts[1]
        calibration = parts[-1]
        model = '_'.join(parts[2:-1]) if len(parts) > 3 else parts[2]
        y_true = df['y_true'].values
        p = df[['p_H', 'p_D', 'p_A']].values
        ece, bins = ece_score(y_true, p)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        if len(bins):
            ax.plot(
                bins['confidence'],
                bins['accuracy'],
                marker='o',
                label=f'{model} ({calibration}) ECE={ece:.3f}'
            )
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        ax.set_xlabel('Mean confidence')
        ax.set_ylabel('Empirical accuracy')
        ax.set_title(f'Reliability: {model} – {calibration}\nTask {task}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'reliability_{task}_{model}_{calibration}.png', dpi=180)
        plt.close(fig)


def build_inplay_minute_metrics(
    out: Path,
    snap_test: pd.DataFrame,
    frozen_probs: np.ndarray,
    test: pd.DataFrame,
    ys_te: np.ndarray,
    class_order: np.ndarray
):
    minute_files = list(out.glob("tmp_minute_L_*.csv"))
    if not minute_files:
        print("No per-minute classification files found; skipping in-play metrics.")
        return
    rows = []
    for f in minute_files:
        df = pd.read_csv(f)
        stem = f.stem
        parts = stem.split('_')
        if len(parts) < 6:
            continue
        model = '_'.join(parts[3:-2])
        calibration = parts[-2]
        minute = int(parts[-1])
        df['model'] = model
        df['calibration'] = calibration
        df['minute'] = minute
        rows.append(df)

    if rows:
        all_minutes = pd.concat(rows, ignore_index=True)
        all_minutes = all_minutes.loc[:, ~all_minutes.columns.str.contains('^Unnamed')]
    else:
        all_minutes = pd.DataFrame()

    frozen_rows = []
    for minute in sorted(snap_test['snapshot_minute'].unique()):
        mask = snap_test['snapshot_minute'] == minute
        match_ids = snap_test.loc[mask, 'match_id'].values
        test_idx = [test[test['match_id'] == mid].index[0] for mid in match_ids]
        probs = frozen_probs[test_idx]
        y_true = ys_te[mask]
        metrics = classification_metrics(y_true, probs)
        metrics['minute'] = minute
        metrics['model'] = 'Frozen_PreMatch_Prior'
        metrics['calibration'] = 'none'
        frozen_rows.append(metrics)

    frozen_df = pd.DataFrame(frozen_rows) if frozen_rows else pd.DataFrame()
    combined = pd.concat([all_minutes, frozen_df], ignore_index=True)
    combined.to_csv(out / 'model3_metrics_by_minute_all_models.csv', index=False)

    if not combined.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (model, cal), group in combined.groupby(['model', 'calibration']):
            label = f"{model} ({cal})" if cal != 'none' else model
            group = group.sort_values('minute')
            ax.plot(group['minute'], group['balanced_accuracy'], marker='o', label=label)
        ax.set_xlabel('Match minute')
        ax.set_ylabel('Balanced accuracy')
        ax.set_title('Task L: Balanced Accuracy vs Minute (all models)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out / 'model3_all_models_balanced_accuracy_vs_minute.png', dpi=180)
        plt.close(fig)

    reg_files = list(out.glob("tmp_minute_L_reg_*.csv"))
    if reg_files:
        reg_rows = []
        for f in reg_files:
            df = pd.read_csv(f)
            stem = f.stem
            parts = stem.split('_')
            if len(parts) < 6:
                continue
            model = '_'.join(parts[3:-2])
            calibration = parts[-2]
            minute = int(parts[-1])
            df['model'] = model
            df['calibration'] = calibration
            df['minute'] = minute
            reg_rows.append(df)
        if reg_rows:
            reg_df = pd.concat(reg_rows, ignore_index=True)
            reg_df.to_csv(out / 'model3_regression_metrics_by_minute_all_models.csv', index=False)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for (model, cal), group in reg_df.groupby(['model', 'calibration']):
                label = f"{model} ({cal})" if cal != 'none' else model
                group = group.sort_values('minute')
                ax2.plot(group['minute'], group['rmse'], marker='s', label=label)
            ax2.set_xlabel('Match minute')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Task L_reg: RMSE vs Minute (all models)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(out / 'model3_regression_rmse_vs_minute_all_models.png', dpi=180)
            plt.close(fig2)


def generate_shap_waterfalls(
    model,
    X,
    indices,
    feature_names,
    output_prefix,
    is_regression=False
):
    explainer = SHAP.TreeExplainer(model)
    shap_values = explainer.shap_values(X[indices])

    if isinstance(shap_values, SHAP.Explanation):
        for i, idx in enumerate(indices):
            exp = shap_values[i]
            SHAP.waterfall_plot(exp, show=False)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_sample_{i+1}_idx_{idx}.png", dpi=150)
            plt.close()
    else:
        expected = explainer.expected_value
        if isinstance(expected, np.ndarray) and expected.size == 1:
            expected = float(expected.item())

        for i, idx in enumerate(indices):
            if is_regression:
                exp = SHAP.Explanation(
                    values=shap_values[i],
                    base_values=expected,
                    data=X[idx],
                    feature_names=feature_names
                )
            else:
                if isinstance(shap_values, list):
                    pred_class = model.predict(X[idx].reshape(1, -1))[0]
                    exp_vals = shap_values[pred_class][i]
                    base_val = expected[pred_class] if isinstance(
                        expected, (list, np.ndarray)
                    ) else expected
                    if isinstance(base_val, np.ndarray) and base_val.size == 1:
                        base_val = float(base_val.item())
                    exp = SHAP.Explanation(
                        values=exp_vals,
                        base_values=base_val,
                        data=X[idx],
                        feature_names=feature_names
                    )
                else:
                    continue

            SHAP.waterfall_plot(exp, show=False)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_sample_{i+1}_idx_{idx}.png", dpi=150)
            plt.close()


def plot_gmm_component_counts(out: Path):
    diag_files = list(out.glob("gmm_diagnostics_*.csv"))
    if not diag_files:
        agg = out / "gmm_diagnostics_all.csv"
        if agg.exists():
            df = pd.read_csv(agg)
        else:
            print("No GMM diagnostics found; skipping component count plot.")
            return
    else:
        dfs = [pd.read_csv(f) for f in diag_files]
        df = pd.concat(dfs, ignore_index=True)

    if df.empty or 'selected_components' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if 'role' in df.columns:
        for role, group in df.groupby('role'):
            ax.hist(
                group['selected_components'],
                alpha=0.5,
                label=role,
                bins=np.arange(0.5, 10, 1)
            )
    else:
        ax.hist(df['selected_components'], bins=np.arange(0.5, 10, 1))
    ax.set_xlabel('Number of selected GMM components')
    ax.set_ylabel('Count')
    ax.set_title('GMM component selection across minority classes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / 'gmm_component_counts.png', dpi=180)
    plt.close(fig)
    summary = df.groupby('role')['selected_components'].describe() if 'role' in df.columns else df['selected_components'].describe()
    summary.to_csv(out / 'gmm_component_summary.csv')


def generate_analysis(pre, snapshots, out: Path, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    train = pre[pre.split == 'train']
    test = pre[pre.split == 'test']
    cols = [
        c for c in pre.columns
        if c not in {
            'match_id', 'kick_off', 'outcome', 'goal_margin', 'split'
        }
        and pd.api.types.is_numeric_dtype(pre[c])
    ]
    Xtr = train[cols].fillna(0).to_numpy(float)
    Xte = test[cols].fillna(0).to_numpy(float)
    ytr = train.outcome.to_numpy()
    yte = test.outcome.to_numpy()

    frozen = train_calibrated('RandomForest', Xtr, ytr, seed)
    frozen_probs = frozen.predict_proba(Xte)
    np.save(out / 'frozen_prematch_test_probs.npy', frozen_probs)
    pd.DataFrame({
        'match_id': test.match_id,
        'p_H': frozen_probs[:, 0],
        'p_D': frozen_probs[:, 1],
        'p_A': frozen_probs[:, 2],
        'outcome': yte
    }).to_csv(out / 'frozen_prematch_test_predictions.csv', index=False)

    raw_rf = classifier_factory('RandomForest', seed)
    raw_rf.fit(Xtr, pd.Categorical(ytr, categories=CLASS_ORDER).codes)
    raw_p = raw_rf.predict_proba(Xte)
    cal_p = frozen_probs
    rows = []
    for label, p in [('before', raw_p), ('after', cal_p)]:
        ece, b = ece_score(yte, p)
        for r in b.to_dict('records'):
            rows.append({'stage': label, 'ECE': ece, **r})
    pd.DataFrame(rows).to_csv(out / 'calibration_reliability_bins.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for label, p, marker in [('before', raw_p, 'o'), ('after', cal_p, 's')]:
        ece, b = ece_score(yte, p)
        if len(b):
            ax.plot(b['confidence'], b['accuracy'], marker=marker, label=f'{label} (ECE={ece:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', label='perfect')
    ax.set(xlabel='Mean confidence', ylabel='Empirical accuracy', title='Calibration: Random Forest, Task C')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'reliability_taskC_rf_before_after.png', dpi=180)
    plt.close(fig)

    snap_train = snapshots[snapshots.split == 'train']
    snap_test = snapshots[snapshots.split == 'test']
    slcols = [
        c for c in snapshots.columns
        if c not in {
            'match_id', 'kick_off', 'outcome', 'goal_margin', 'split'
        }
        and pd.api.types.is_numeric_dtype(snapshots[c])
    ]
    Xs_tr = snap_train[slcols].fillna(0).to_numpy(float)
    Xs_te = snap_test[slcols].fillna(0).to_numpy(float)
    ys_tr = snap_train.outcome.to_numpy()
    ys_te = snap_test.outcome.to_numpy()

    live = train_calibrated('RandomForest', Xs_tr, ys_tr, seed)
    livep = live.predict_proba(Xs_te)
    build_inplay_minute_metrics(out, snap_test, frozen_probs, test, ys_te, CLASS_ORDER)

    phase_bins = []
    phases = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 91)]
    for lo, hi in phases:
        mask = (snap_test.snapshot_minute.to_numpy() >= lo) & (snap_test.snapshot_minute.to_numpy() < hi)
        ece, b = ece_score(ys_te[mask], livep[mask])
        phase_bins.append({
            'phase': f'{lo}-{min(90, hi)}',
            'n': int(mask.sum()),
            'ECE': ece,
            'mean_confidence': float(livep[mask].max(1).mean()),
            'accuracy': float((CLASS_ORDER[livep[mask].argmax(1)] == ys_te[mask]).mean())
        })
    pd.DataFrame(phase_bins).to_csv(out / 'model3_calibration_by_phase.csv', index=False)

    pred_L = CLASS_ORDER[livep.argmax(1)]
    true_L = ys_te
    eps = 1e-9
    true_index_L = pd.Categorical(true_L, categories=CLASS_ORDER).codes
    losses_L = -np.log(np.clip(livep[np.arange(len(true_L)), true_index_L], eps, 1))
    worst_idx_L = np.argsort(losses_L)[-10:][::-1]
    worst_L = snap_test.iloc[worst_idx_L].copy()
    worst_L['loss'] = losses_L[worst_idx_L]
    worst_L['predicted'] = pred_L[worst_idx_L]
    worst_L.to_csv(out / 'worst10_taskL_randomforest.csv', index=False)

    true_index_C = pd.Categorical(test.outcome, categories=CLASS_ORDER).codes
    losses_C = -np.log(np.clip(frozen_probs[np.arange(len(test)), true_index_C], eps, 1))
    worst_idx_C = np.argsort(losses_C)[-10:][::-1]
    worst_C = test.iloc[worst_idx_C].copy()
    worst_C['loss'] = losses_C[worst_idx_C]
    worst_C['predicted'] = CLASS_ORDER[frozen_probs[worst_idx_C].argmax(1)]
    for i, c in enumerate(CLASS_ORDER):
        worst_C[f'p_{c}'] = frozen_probs[worst_idx_C, i]
    worst_C.to_csv(out / 'worst10_taskC_randomforest.csv', index=False)

    pred_R_file = out.parent / 'phase2' / 'predictions_R_RandomForest_none.csv'
    if pred_R_file.exists():
        pred_df = pd.read_csv(pred_R_file)
        test_R = pre[pre.split == 'test'].reset_index(drop=True).copy()
        test_R['prediction'] = pred_df['prediction'].values
        test_R['abs_error'] = np.abs(test_R['goal_margin'] - test_R['prediction'])
        worst_idx_R = np.argsort(test_R['abs_error'])[-10:][::-1]
        worst_R = test_R.iloc[worst_idx_R].copy()
        worst_R.to_csv(out / 'worst10_taskR_randomforest.csv', index=False)
    else:
        reg_model = RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        )
        reg_model.fit(Xtr, train['goal_margin'].to_numpy(float))
        pred_R = reg_model.predict(Xte)
        test_R = test.copy()
        test_R['prediction'] = pred_R
        test_R['abs_error'] = np.abs(test_R['goal_margin'] - pred_R)
        worst_idx_R = np.argsort(test_R['abs_error'])[-10:][::-1]
        worst_R = test_R.iloc[worst_idx_R].copy()
        worst_R.to_csv(out / 'worst10_taskR_randomforest.csv', index=False)
        reg_model_for_shap = reg_model

    print("Generating SHAP waterfalls for Task C worst predictions...")
    generate_shap_waterfalls(
        raw_rf,
        Xte,
        worst_idx_C,
        cols,
        str(out / 'shap_waterfall_taskC_worst'),
        is_regression=False
    )

    live_raw = classifier_factory('RandomForest', seed)
    live_raw.fit(Xs_tr, pd.Categorical(ys_tr, categories=CLASS_ORDER).codes)
    print("Generating SHAP waterfalls for Task L worst predictions...")
    generate_shap_waterfalls(
        live_raw,
        Xs_te,
        worst_idx_L,
        slcols,
        str(out / 'shap_waterfall_taskL_worst'),
        is_regression=False
    )

    if pred_R_file.exists():
        reg_model = RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        )
        reg_model.fit(Xtr, train['goal_margin'].to_numpy(float))
    else:
        reg_model = reg_model_for_shap
    print("Generating SHAP waterfalls for Task R worst predictions...")
    generate_shap_waterfalls(
        reg_model,
        Xte,
        worst_idx_R,
        cols,
        str(out / 'shap_waterfall_taskR_worst'),
        is_regression=True
    )

    Xshow = Xte[:min(63, len(Xte))]
    for name in ['RandomForest', 'GBM', 'XGBoost', 'LightGBM']:
        try:
            model = classifier_factory(name, seed)
            model.fit(Xtr, pd.Categorical(ytr, categories=CLASS_ORDER).codes)
            explainer = SHAP.TreeExplainer(model)
            sv = explainer.shap_values(Xshow)
            if isinstance(sv, list):
                arr = np.mean(
                    np.abs(np.stack([np.asarray(x) for x in sv], axis=0)),
                    axis=(0, 2)
                )
            else:
                a = np.asarray(sv)
                arr = np.mean(np.abs(a), axis=(0, 2)) if a.ndim == 3 else np.mean(np.abs(a), axis=0)
            order = np.argsort(arr)[-12:][::-1]
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            ax.barh(np.arange(len(order)), arr[order])
            ax.set_yticks(np.arange(len(order)))
            ax.set_yticklabels(np.array(cols)[order])
            ax.invert_yaxis()
            ax.set_title(f'SHAP global importance - {name} - Task C')
            ax.set_xlabel('mean(|SHAP|)')
            fig.tight_layout()
            fig.savefig(out / f'shap_global_C_{name}.png', dpi=180)
            plt.close(fig)
            pd.DataFrame({
                'feature': cols,
                'mean_abs_shap': arr
            }).sort_values('mean_abs_shap', ascending=False).to_csv(
                out / f'shap_global_C_{name}.csv', index=False
            )
        except Exception as model_exc:
            (out / f'shap_error_{name}.txt').write_text(str(model_exc), encoding='utf-8')

    ids = snap_test.match_id.unique()
    if len(ids) > 0:
        chosen = int(ids[0])
        mask = snap_test.match_id.to_numpy() == chosen
        Xmatch = snap_test.loc[mask, slcols].fillna(0).to_numpy(float)
        live_raw_timeline = classifier_factory('RandomForest', seed)
        live_raw_timeline.fit(Xs_tr, pd.Categorical(ys_tr, categories=CLASS_ORDER).codes)
        exp = SHAP.TreeExplainer(live_raw_timeline)
        sv = exp.shap_values(Xmatch)
        if isinstance(sv, list):
            vals = np.mean(
                np.abs(np.stack([np.asarray(v) for v in sv], axis=0)),
                axis=(0, 2)
            )
        else:
            aa = np.asarray(sv)
            vals = np.mean(np.abs(aa), axis=(0, 2)) if aa.ndim == 3 else np.mean(np.abs(aa), axis=0)
        top = np.argsort(vals)[-6:][::-1]
        arr = np.asarray(sv)
        if isinstance(sv, list):
            arr = np.stack([np.asarray(v) for v in sv], axis=-1)
        if arr.ndim == 3:
            timeline = np.mean(np.abs(arr[:, top, :]), axis=2)
        else:
            timeline = np.abs(arr[:, top]).mean(axis=1)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for j, fidx in enumerate(top):
            ax.plot(
                snap_test.loc[mask, 'snapshot_minute'],
                timeline[:, j],
                marker='o',
                label=slcols[fidx]
            )
        ax.set(xlabel='Match minute', ylabel='Mean |SHAP| contribution', title=f'SHAP timeline - Model 3 match {chosen}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'shap_inplay_timeline.png', dpi=180)
        plt.close(fig)

    print("Running kernel scaling experiment (SVC on increasing subset sizes)...")
    max_n = min(len(Xtr), 800)
    if max_n < 10:
        sizes = [max(1, max_n // 4), max_n // 2, max_n]
        sizes = sorted(set(sizes))
    else:
        base_sizes = [10, 20, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
        sizes = [s for s in base_sizes if s <= max_n]
        if len(sizes) < 4:
            fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
            extra = [int(f * max_n) for f in fractions if int(f * max_n) not in sizes and int(f * max_n) > 0]
            sizes = sorted(set(sizes + extra))
    if len(sizes) < 3:
        sizes = list(np.linspace(1, max_n, min(max_n, 3), dtype=int))
    sizes = sorted(set(sizes))
    print(f"Subset sizes: {sizes}")
    scale_rows = []
    rng = np.random.default_rng(seed)
    for n in sizes:
        idx = rng.choice(max_n, size=n, replace=False)
        X_sub = Xtr[idx]
        y_sub_enc = pd.Categorical(ytr[idx], categories=CLASS_ORDER).codes

        if len(np.unique(y_sub_enc)) < 2:
            print(f"  n={n}, skipped (subset contains only 1 class)")
            continue

        model = SVC(C=1.0, gamma='scale', probability=False, random_state=seed)
        t0 = time.perf_counter()
        model.fit(X_sub, y_sub_enc)
        elapsed = time.perf_counter() - t0
        scale_rows.append({'n': n, 'seconds': elapsed})
        print(f"  n={n}, time={elapsed:.4f}s")
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(out / 'kernel_scaling.csv', index=False)
    if not scale_df.empty:
        fig, ax = plt.subplots(figsize=(6.7, 5.0))
        ax.plot(scale_df['n'], scale_df['seconds'], marker='o', linestyle='-', color='b')
        ax.set(xlabel='Training samples (n)', ylabel='Fit time (s)', title='Empirical Kernel-SVM Scaling (Real Data)')
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()
        fig.savefig(out / 'kernel_scaling.png', dpi=180)
        plt.close(fig)
        print("Kernel scaling plot saved.")
    else:
        print("No data points for kernel scaling plot – skipping.")

    plot_gmm_component_counts(out)
    joblib.dump(frozen, out / 'model1_randomforest_calibrated.joblib')
    joblib.dump(live, out / 'model3_randomforest_calibrated.joblib')
    generate_reliability_plots(out)
    print("All analysis completed.")