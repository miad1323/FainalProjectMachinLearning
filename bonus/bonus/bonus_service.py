from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

P1_OUT = PROJECT_ROOT / "phase1" / "outputs"
P2_OUT = PROJECT_ROOT / "phase2" / "outputs"
PREMATCH_PATH = P1_OUT / "pre_match_features_final.csv"
SNAPSHOT_PATH = P1_OUT / "in_play_snapshots_all_seasons.csv"
FEATURES_PATH = P2_OUT / "deployment_features.json"
PRE_CLF_PATH = P2_OUT / "deployment_prematch_classifier.joblib"
PRE_REG_PATH = P2_OUT / "deployment_prematch_regressor.joblib"
LIVE_CLF_PATH = P2_OUT / "deployment_inplay_classifier.joblib"
LIVE_REG_PATH = P2_OUT / "deployment_inplay_regressor.joblib"

CLASS_ORDER = ["H", "D", "A"]

try:
    import shap
except Exception:
    shap = None


def _require_artifacts() -> None:
    required = [PREMATCH_PATH, SNAPSHOT_PATH, FEATURES_PATH, PRE_CLF_PATH, PRE_REG_PATH, LIVE_CLF_PATH, LIVE_REG_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Final deployment artifacts are missing. Run phase2/phase2_final.ipynb first. Missing: "
            + ", ".join(missing)
        )


_require_artifacts()
FEATURES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
PRE_FEATURES = list(FEATURES["prematch"])
LIVE_FEATURES = list(FEATURES["inplay"])

def _repair_loaded_estimator(model) -> int:
    """Repair safe fitted-state drift in sklearn pickles across minor versions.

    scikit-learn 1.8 reads ``SimpleImputer._fill_dtype`` during transform, while
    older fitted artifacts may not contain it.  The value is fully determined by
    the fitted statistics dtype, so restoring it preserves the original behavior.
    This is a compatibility guard only; it does not refit or change predictions.
    """
    repaired = 0
    seen: set[int] = set()

    def walk(obj) -> None:
        nonlocal repaired
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        if obj.__class__.__name__ == "SimpleImputer" and hasattr(obj, "statistics_") and not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = np.asarray(obj.statistics_).dtype
            repaired += 1

        # Some LogisticRegression pickles created under a different sklearn minor
        # release can miss this fitted/config attribute when loaded later.
        if obj.__class__.__name__ == "LogisticRegression" and not hasattr(obj, "multi_class"):
            obj.multi_class = "auto"
            repaired += 1

        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                walk(step)
        if hasattr(obj, "transformers_"):
            for item in obj.transformers_:
                if len(item) >= 2:
                    walk(item[1])
        if hasattr(obj, "estimators_"):
            estimators = obj.estimators_
            if isinstance(estimators, dict):
                estimators = estimators.values()
            for est in estimators:
                walk(est)

    walk(model)
    return repaired


PRE_CLF = joblib.load(PRE_CLF_PATH)
PRE_REG = joblib.load(PRE_REG_PATH)
LIVE_CLF = joblib.load(LIVE_CLF_PATH)
LIVE_REG = joblib.load(LIVE_REG_PATH)
COMPAT_REPAIRS = sum(_repair_loaded_estimator(m) for m in [PRE_CLF, PRE_REG, LIVE_CLF, LIVE_REG])

PRE = pd.read_csv(PREMATCH_PATH)
PRE["match_id"] = PRE["match_id"].astype(int)
PRE_INDEX = PRE.set_index("match_id", drop=False)

SNAP = pd.read_csv(SNAPSHOT_PATH)
SNAP["match_id"] = SNAP["match_id"].astype(int)
# Add the final compact pre-match features to every live snapshot. This is exactly
# the feature contract used when the final Model 3 was fitted.
SNAP = SNAP.merge(PRE[["match_id", *PRE_FEATURES]], on="match_id", how="left", validate="many_to_one")
SNAP = SNAP.sort_values(["match_id", "snapshot_minute"]).reset_index(drop=True)

# Precompute vectors once at startup: no CSV scans or feature reassembly inside the
# request path. This is the main latency engineering change requested by the TA.
PRE_VECTOR: dict[int, np.ndarray] = {
    int(r.match_id): np.asarray([pd.to_numeric(r.get(c), errors="coerce") for c in PRE_FEATURES], dtype=float).reshape(1, -1)
    for _, r in PRE.iterrows()
}
SNAP_VECTOR: dict[tuple[int, int], np.ndarray] = {}
MINUTES_BY_MATCH: dict[int, np.ndarray] = {}
for mid, part in SNAP.groupby("match_id", sort=False):
    minutes = part["snapshot_minute"].astype(int).to_numpy()
    MINUTES_BY_MATCH[int(mid)] = minutes
    for _, r in part.iterrows():
        minute = int(r["snapshot_minute"])
        vec = np.asarray([pd.to_numeric(r.get(c), errors="coerce") for c in LIVE_FEATURES], dtype=float).reshape(1, -1)
        SNAP_VECTOR[(int(mid), minute)] = vec


def _align_probs(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)[0]
    out = np.zeros(3, dtype=float)
    for j, c in enumerate(model.classes_):
        out[CLASS_ORDER.index(str(c))] = raw[j]
    return out


def _nearest_minute(match_id: int, requested: int) -> int:
    minutes = MINUTES_BY_MATCH.get(int(match_id))
    if minutes is None or len(minutes) == 0:
        raise ValueError(f"No real snapshots for match {match_id}")
    pos = np.searchsorted(minutes, requested, side="right") - 1
    pos = max(0, min(pos, len(minutes) - 1))
    return int(minutes[pos])


def _collapse_shap(values, n_features: int) -> np.ndarray:
    # SHAP versions differ for multiclass estimators. Collapse every non-feature
    # dimension by mean absolute value and always return exactly n_features values.
    arr = np.asarray(getattr(values, "values", values), dtype=float)
    arr = np.abs(arr)
    if arr.ndim == 1:
        vec = arr
    else:
        feature_axes = [i for i, size in enumerate(arr.shape) if size == n_features]
        faxis = feature_axes[0] if feature_axes else (arr.ndim - 1)
        arr = np.moveaxis(arr, faxis, -1)
        vec = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    if len(vec) != n_features:
        vec = np.resize(vec, n_features)
    return vec


def _pipeline_transformed(pipeline, X: np.ndarray) -> tuple[object, np.ndarray]:
    transformed = X
    # Transform through every step except the final estimator.
    for _, step in pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    return pipeline.steps[-1][1], np.asarray(transformed, dtype=float)


# Explainers are constructed exactly once. Prediction remains available even if SHAP
# is not installed, but the final zip pins it in requirements.txt.
PRE_EXPLAINER = LIVE_EXPLAINER = None
if shap is not None:
    try:
        pre_est, pre_bg = _pipeline_transformed(PRE_CLF, PRE[PRE_FEATURES].to_numpy(float))
        if hasattr(pre_est, "coef_"):
            PRE_EXPLAINER = shap.LinearExplainer(pre_est, pre_bg)
    except Exception:
        PRE_EXPLAINER = None
    try:
        live_est, live_bg = _pipeline_transformed(LIVE_CLF, SNAP[LIVE_FEATURES].to_numpy(float))
        if hasattr(live_est, "coef_"):
            LIVE_EXPLAINER = shap.LinearExplainer(live_est, live_bg)
        else:
            LIVE_EXPLAINER = shap.TreeExplainer(live_est)
    except Exception:
        LIVE_EXPLAINER = None


def _top_shap(model_kind: str, X: np.ndarray, feature_names: list[str]) -> Optional[Dict[str, float]]:
    explainer = PRE_EXPLAINER if model_kind == "pre_match" else LIVE_EXPLAINER
    pipeline = PRE_CLF if model_kind == "pre_match" else LIVE_CLF
    if explainer is None:
        return None
    try:
        _, transformed = _pipeline_transformed(pipeline, X)
        values = explainer(transformed)
        vec = _collapse_shap(values, len(feature_names))
        top = np.argsort(vec)[-5:][::-1]
        return {feature_names[int(i)]: float(vec[int(i)]) for i in top}
    except Exception:
        return None


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


@lru_cache(maxsize=4096)
def _predict_cached(match_id: int, minute: int) -> dict:
    if minute == 0:
        X = PRE_VECTOR.get(int(match_id))
        if X is None:
            raise ValueError(f"No real pre-match row for match {match_id}")
        p = _align_probs(PRE_CLF, X)
        margin = float(np.clip(PRE_REG.predict(X)[0], -5, 5))
        top = _top_shap("pre_match", X, PRE_FEATURES)
        return {
            "match_id": int(match_id), "snapshot_minute": 0, "actual_minute": None,
            "model_used": "pre_match", "probabilities": dict(zip(CLASS_ORDER, map(float, p))),
            "expected_margin": margin, "top_shap": top,
        }

    actual = _nearest_minute(int(match_id), int(minute))
    X = SNAP_VECTOR[(int(match_id), actual)]
    p = _align_probs(LIVE_CLF, X)
    margin = float(np.clip(LIVE_REG.predict(X)[0], -5, 5))
    top = _top_shap("in_play", X, LIVE_FEATURES)
    return {
        "match_id": int(match_id), "snapshot_minute": int(minute), "actual_minute": actual,
        "model_used": "in_play", "probabilities": dict(zip(CLASS_ORDER, map(float, p))),
        "expected_margin": margin, "top_shap": top,
    }


app = FastAPI(title="Football Real-Time Prediction Service", version="2.1-dashboard")

# The dashboard is served by the SAME FastAPI process. This avoids file:// CORS
# problems and means the browser always talks to the exact port that is alive.
DASHBOARD_PATH = Path(__file__).resolve().parent / "football_ml_dashboard.html"
LATENCY_SUMMARY_PATH = Path(__file__).resolve().parent / "outputs" / "latency_summary_final.json"

# CORS is still enabled for debugging or alternate front-ends, although the final
# dashboard is same-origin and therefore does not depend on CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return FileResponse(DASHBOARD_PATH, media_type="text/html")
    return HTMLResponse(
        "<h2>Dashboard file missing</h2><p>Expected bonus/football_ml_dashboard.html</p>",
        status_code=404,
    )


@app.get("/dashboard", include_in_schema=False)
async def dashboard_alias():
    return await dashboard()


@app.get("/metrics")
async def metrics():
    if LATENCY_SUMMARY_PATH.exists():
        try:
            payload = json.loads(LATENCY_SUMMARY_PATH.read_text(encoding="utf-8"))
            payload["available"] = True
            return payload
        except Exception as exc:
            return {"available": False, "error": str(exc)}
    return {
        "available": False,
        "message": "Run the 200-request benchmark in bonus_final.ipynb first."
    }



@app.post("/predict", response_model=PredictionResponse)
async def predict(request: SnapshotRequest):
    try:
        return PredictionResponse(**_predict_cached(int(request.match_id), int(request.snapshot_minute)))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "real_data_only": True,
        "prematch_model": type(PRE_CLF.steps[-1][1]).__name__ if hasattr(PRE_CLF, "steps") else type(PRE_CLF).__name__,
        "inplay_model": type(LIVE_CLF.steps[-1][1]).__name__ if hasattr(LIVE_CLF, "steps") else type(LIVE_CLF).__name__,
        "prematch_feature_count": len(PRE_FEATURES),
        "inplay_feature_count": len(LIVE_FEATURES),
        "cached_prematch_vectors": len(PRE_VECTOR),
        "cached_snapshot_vectors": len(SNAP_VECTOR),
        "shap_available": bool(PRE_EXPLAINER is not None or LIVE_EXPLAINER is not None),
        "demo_match_id": int(PRE["match_id"].iloc[-1]) if len(PRE) else None,
    }
