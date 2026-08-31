from __future__ import annotations
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import joblib
except ImportError:
    joblib = None

try:
    import shap
    SHAP = shap
except ImportError:
    SHAP = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PREMATCH_PATH = PROJECT_ROOT / "phase1" / "outputs" / "pre_match_features_all_seasons.csv"
SNAPSHOT_PATH = PROJECT_ROOT / "phase1" / "outputs" / "in_play_snapshots_all_seasons.csv"
MODEL_DIR = PROJECT_ROOT / "bonus" / "outputs"

clf_model1 = reg_model1 = clf_uncalibrated1 = feature_cols1 = scaler1 = None
clf_model3 = reg_model3 = clf_uncalibrated3 = feature_cols3 = scaler3 = None
prematch_df = snapshots_df = None
prematch_cache: Dict[int, np.ndarray] = {}


def load_data():
    global prematch_df, snapshots_df
    if prematch_df is None:
        if not PREMATCH_PATH.exists():
            raise FileNotFoundError(f"Pre-match data not found at {PREMATCH_PATH}")
        prematch_df = pd.read_csv(PREMATCH_PATH)
        prematch_df["match_id"] = prematch_df["match_id"].astype(int)

    if snapshots_df is None:
        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(f"Snapshot data not found at {SNAPSHOT_PATH}")
        snapshots_df = pd.read_csv(SNAPSHOT_PATH)
        snapshots_df["match_id"] = snapshots_df["match_id"].astype(int)
        snapshots_df.set_index(["match_id", "snapshot_minute"], inplace=True, drop=False)


def get_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def _extract_base_estimator(calibrated):
    try:
        cc = calibrated.calibrated_classifiers_[0]
        return getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
    except Exception:
        return None


def load_or_train_models():
    global clf_model1, reg_model1, clf_uncalibrated1, feature_cols1, scaler1
    global clf_model3, reg_model3, clf_uncalibrated3, feature_cols3, scaler3

    if joblib is None:
        print("ERROR: joblib not installed.")
        return

    model_dir = get_model_dir()

    clf1_path = model_dir / "model1_randomforest_calibrated.joblib"
    reg1_path = model_dir / "model1_randomforest_regressor.joblib"
    uncal1_path = model_dir / "model1_randomforest_uncalibrated.joblib"
    fc1_path = model_dir / "feature_columns_model1.json"
    scaler1_path = model_dir / "scaler_model1.joblib"

    clf3_path = model_dir / "model3_randomforest_calibrated.joblib"
    reg3_path = model_dir / "model3_randomforest_regressor.joblib"
    uncal3_path = model_dir / "model3_randomforest_uncalibrated.joblib"
    fc3_path = model_dir / "feature_columns_model3.json"
    scaler3_path = model_dir / "scaler_model3.joblib"

    model1_loaded = (
        clf1_path.exists()
        and reg1_path.exists()
        and fc1_path.exists()
        and scaler1_path.exists()
    )
    model3_loaded = (
        clf3_path.exists()
        and reg3_path.exists()
        and fc3_path.exists()
        and scaler3_path.exists()
    )

    if not (model1_loaded and model3_loaded):
        print("One or both models missing – training from scratch...")
        train_all_models(model_dir)
        model1_loaded = clf1_path.exists()
        model3_loaded = clf3_path.exists()

    if model1_loaded:
        clf_model1 = joblib.load(clf1_path)
        reg_model1 = joblib.load(reg1_path)
        clf_uncalibrated1 = joblib.load(uncal1_path) if uncal1_path.exists() else None
        with open(fc1_path, "r") as f:
            feature_cols1 = json.load(f)
        scaler1 = joblib.load(scaler1_path)

    if model3_loaded:
        clf_model3 = joblib.load(clf3_path)
        reg_model3 = joblib.load(reg3_path)
        clf_uncalibrated3 = joblib.load(uncal3_path) if uncal3_path.exists() else None
        with open(fc3_path, "r") as f:
            feature_cols3 = json.load(f)
        scaler3 = joblib.load(scaler3_path)

    if clf_uncalibrated1 is None and clf_model1 is not None:
        clf_uncalibrated1 = _extract_base_estimator(clf_model1)
    if clf_uncalibrated3 is None and clf_model3 is not None:
        clf_uncalibrated3 = _extract_base_estimator(clf_model3)


def train_all_models(model_dir: Path):
    global feature_cols1, scaler1, feature_cols3, scaler3
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    pre = pd.read_csv(PREMATCH_PATH)
    exclude = {
        "match_id", "kick_off", "season", "home_team", "away_team",
        "outcome", "goal_margin", "split"
    }
    feature_cols1 = [
        c for c in pre.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(pre[c])
    ]

    X1 = pre[feature_cols1].fillna(0.0).to_numpy(float)
    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)

    base_clf1 = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    base_clf1.fit(X1_scaled, pre["outcome"].to_numpy())
    clf1 = CalibratedClassifierCV(base_clf1, method='sigmoid', cv=3, ensemble=True)
    clf1.fit(X1_scaled, pre["outcome"].to_numpy())
    reg1 = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    reg1.fit(X1_scaled, pre["goal_margin"].to_numpy(float))

    joblib.dump(base_clf1, model_dir / "model1_randomforest_uncalibrated.joblib")
    joblib.dump(clf1, model_dir / "model1_randomforest_calibrated.joblib")
    joblib.dump(reg1, model_dir / "model1_randomforest_regressor.joblib")
    joblib.dump(scaler1, model_dir / "scaler_model1.joblib")
    with open(model_dir / "feature_columns_model1.json", "w") as f:
        json.dump(feature_cols1, f)

    snap = pd.read_csv(SNAPSHOT_PATH)
    train_snap = (
        snap[snap.split == "train"].copy()
        if "split" in snap.columns and not snap[snap.split == "train"].empty
        else snap.copy()
    )
    exclude3 = {"match_id", "kick_off", "outcome", "goal_margin", "split"}
    feature_cols3 = [
        c for c in train_snap.columns
        if c not in exclude3
        and not str(c).startswith("prematch_")
        and pd.api.types.is_numeric_dtype(train_snap[c])
    ]

    X3 = train_snap[feature_cols3].fillna(0.0).to_numpy(float)
    scaler3 = StandardScaler()
    X3_scaled = scaler3.fit_transform(X3)

    base_clf3 = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    base_clf3.fit(X3_scaled, train_snap["outcome"].to_numpy())
    clf3 = CalibratedClassifierCV(base_clf3, method='sigmoid', cv=3, ensemble=True)
    clf3.fit(X3_scaled, train_snap["outcome"].to_numpy())
    reg3 = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    reg3.fit(X3_scaled, train_snap["goal_margin"].to_numpy(float))

    joblib.dump(base_clf3, model_dir / "model3_randomforest_uncalibrated.joblib")
    joblib.dump(clf3, model_dir / "model3_randomforest_calibrated.joblib")
    joblib.dump(reg3, model_dir / "model3_randomforest_regressor.joblib")
    joblib.dump(scaler3, model_dir / "scaler_model3.joblib")
    with open(model_dir / "feature_columns_model3.json", "w") as f:
        json.dump(feature_cols3, f)


def clean_value(val):
    if pd.isna(val):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def assemble_prematch_features(match_id: int) -> np.ndarray:
    global prematch_df, feature_cols1, scaler1, prematch_cache
    if match_id in prematch_cache:
        return prematch_cache[match_id]

    row = prematch_df[prematch_df["match_id"] == match_id]
    if row.empty:
        raise ValueError(f"No pre-match data for match {match_id}")

    vector = [clean_value(row.iloc[0].get(col, 0.0)) for col in feature_cols1]
    X = np.nan_to_num(
        np.array(vector, dtype=float).reshape(1, -1),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    if scaler1 is not None:
        X = scaler1.transform(X)

    prematch_cache[match_id] = X.flatten()
    return prematch_cache[match_id]


def get_nearest_snapshot(match_id: int, requested_minute: int) -> Tuple[int, pd.Series]:
    try:
        match_rows = snapshots_df.xs(match_id, level=0, drop_level=False)
    except KeyError:
        raise ValueError(f"No snapshots for match {match_id}")

    minutes = match_rows.index.get_level_values('snapshot_minute')
    pos = max(0, minutes.searchsorted(requested_minute, side='right') - 1)
    actual_minute = int(minutes[pos])

    row = match_rows.loc[(match_id, actual_minute)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return actual_minute, row


def assemble_inplay_features(match_id: int, requested_minute: int) -> Tuple[int, np.ndarray]:
    global prematch_df, feature_cols3, scaler3

    prematch_row = prematch_df[prematch_df["match_id"] == match_id]
    if prematch_row.empty:
        raise ValueError(f"No pre-match data for match {match_id}")

    prematch_dict = {
        k: v for k, v in prematch_row.iloc[0].to_dict().items()
        if k not in ["match_id", "kick_off", "season", "home_team", "away_team",
                     "outcome", "goal_margin", "split"]
    }
    actual_minute, snapshot_row = get_nearest_snapshot(match_id, requested_minute)
    snapshot_dict = {
        k: v for k, v in snapshot_row.to_dict().items()
        if k not in ["match_id", "kick_off", "outcome", "goal_margin", "split"]
    }

    vector = []
    for col in feature_cols3:
        val = snapshot_dict.get(col, prematch_dict.get(col, 0.0))
        vector.append(clean_value(val))

    X = np.nan_to_num(
        np.array(vector, dtype=float).reshape(1, -1),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    if scaler3 is not None:
        X = scaler3.transform(X)

    return actual_minute, X.flatten()


load_data()
load_or_train_models()

app = FastAPI(title="Football Live Prediction Service (Bonus)", version="1.0")


class SnapshotRequest(BaseModel):
    match_id: int
    snapshot_minute: int = Field(..., ge=0, le=90)


class PredictionResponse(BaseModel):
    match_id: int
    snapshot_minute: int
    actual_minute: Optional[int] = None
    model_used: str
    probabilities: Dict[str, float]
    expected_margin: float
    top_shap: Optional[Dict[str, float]] = None


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SnapshotRequest):
    minute = request.snapshot_minute
    try:
        if minute == 0:
            if clf_model1 is None or reg_model1 is None:
                raise HTTPException(status_code=503, detail="Pre-match models not available.")
            X = assemble_prematch_features(request.match_id).reshape(1, -1)
            probs = clf_model1.predict_proba(X)[0] if hasattr(clf_model1, "predict_proba") else [1/3, 1/3, 1/3]
            margin = float(np.clip(reg_model1.predict(X)[0], -5, 5))

            if hasattr(clf_model1, "classes_") and list(clf_model1.classes_) != ["H", "D", "A"]:
                idx = [list(clf_model1.classes_).index(c) for c in ["H", "D", "A"]]
                probs = probs[idx]

            top_shap = None
            if clf_uncalibrated1 and SHAP:
                try:
                    explainer = SHAP.TreeExplainer(clf_uncalibrated1)
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    else:
                        shap_abs = np.abs(shap_values)
                    if shap_abs.ndim == 2:
                        shap_abs = shap_abs[0]
                    top_indices = np.argsort(shap_abs)[-5:][::-1]
                    top_shap = {feature_cols1[i]: float(shap_abs[i]) for i in top_indices}
                except Exception as e:
                    print(f"SHAP failed: {e}")

            return PredictionResponse(
                match_id=request.match_id,
                snapshot_minute=minute,
                actual_minute=None,
                model_used="pre_match",
                probabilities={"H": probs[0], "D": probs[1], "A": probs[2]},
                expected_margin=margin,
                top_shap=top_shap
            )
        else:
            if clf_model3 is None or reg_model3 is None:
                raise HTTPException(status_code=503, detail="In-play models not available.")
            actual_minute, X = assemble_inplay_features(request.match_id, minute)
            X = X.reshape(1, -1)
            probs = clf_model3.predict_proba(X)[0] if hasattr(clf_model3, "predict_proba") else [1/3, 1/3, 1/3]
            margin = float(np.clip(reg_model3.predict(X)[0], -5, 5))

            if hasattr(clf_model3, "classes_") and list(clf_model3.classes_) != ["H", "D", "A"]:
                idx = [list(clf_model3.classes_).index(c) for c in ["H", "D", "A"]]
                probs = probs[idx]

            top_shap = None
            if clf_uncalibrated3 and SHAP:
                try:
                    explainer = SHAP.TreeExplainer(clf_uncalibrated3)
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    else:
                        shap_abs = np.abs(shap_values)
                    if shap_abs.ndim == 2:
                        shap_abs = shap_abs[0]
                    top_indices = np.argsort(shap_abs)[-5:][::-1]
                    top_shap = {feature_cols3[i]: float(shap_abs[i]) for i in top_indices}
                except Exception as e:
                    print(f"SHAP failed: {e}")

            return PredictionResponse(
                match_id=request.match_id,
                snapshot_minute=minute,
                actual_minute=actual_minute,
                model_used="in_play",
                probabilities={"H": probs[0], "D": probs[1], "A": probs[2]},
                expected_margin=margin,
                top_shap=top_shap
            )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": all([clf_model1, reg_model1, clf_model3, reg_model3])}