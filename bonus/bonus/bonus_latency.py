from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "bonus" / "outputs"


def benchmark(base_url: str = "http://127.0.0.1:8001", n_requests: int = 200, warmup: int = 20, timeout: float = 5.0):
    pre = pd.read_csv(PROJECT_ROOT / "phase1" / "outputs" / "pre_match_features_final.csv")
    match_ids = pre.loc[pre["season"].eq("2020/2021"), "match_id"].astype(int).tolist()
    if not match_ids:
        match_ids = pre["match_id"].astype(int).tolist()
    minutes = [0, 15, 30, 45, 60, 75, 90]
    payloads = [
        {"match_id": match_ids[i % len(match_ids)], "snapshot_minute": minutes[i % len(minutes)]}
        for i in range(n_requests + warmup)
    ]

    session = requests.Session()
    for payload in payloads[:warmup]:
        r = session.post(f"{base_url}/predict", json=payload, timeout=timeout)
        r.raise_for_status()

    rows = []
    for i, payload in enumerate(payloads[warmup:], start=1):
        start = time.perf_counter_ns()
        ok = False
        status = None
        try:
            r = session.post(f"{base_url}/predict", json=payload, timeout=timeout)
            status = r.status_code
            r.raise_for_status()
            ok = True
        except Exception:
            ok = False
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        rows.append({"request": i, **payload, "latency_ms": latency_ms, "ok": ok, "status_code": status})

    detail = pd.DataFrame(rows)
    success = detail.loc[detail["ok"], "latency_ms"].to_numpy(float)
    if len(success) == 0:
        raise RuntimeError("Latency benchmark had zero successful requests")
    summary = {
        "n_requests": int(n_requests),
        "successful": int(detail["ok"].sum()),
        "error_rate": float(1.0 - detail["ok"].mean()),
        "mean_ms": float(np.mean(success)),
        "p50_ms": float(np.percentile(success, 50)),
        "p95_ms": float(np.percentile(success, 95)),
        "p99_ms": float(np.percentile(success, 99)),
        "max_ms": float(np.max(success)),
        "requirement_ms": 200.0,
        "pass_p95_under_200ms": bool(np.percentile(success, 95) < 200.0),
        "pass_p99_under_200ms": bool(np.percentile(success, 99) < 200.0),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_DIR / "latency_requests_final.csv", index=False)
    (OUTPUT_DIR / "latency_summary_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "latency_summary_final.csv", index=False)

    # Presentation-ready latency evidence. This is generated from the same 200
    # measured HTTP requests used for the p50/p95/p99 table.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = min(20, max(8, int(np.sqrt(len(success)))))
        ax.hist(success, bins=bins, alpha=0.9)
        p95 = float(np.percentile(success, 95))
        p99 = float(np.percentile(success, 99))
        ax.axvline(p95, linestyle="--", linewidth=2, label=f"p95 = {p95:.2f} ms")
        ax.axvline(p99, linestyle=":", linewidth=2, label=f"p99 = {p99:.2f} ms")
        ax.set_title("FastAPI latency: 200 measured requests after warm-up")
        ax.set_xlabel("End-to-end HTTP latency (ms)")
        ax.set_ylabel("Requests")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "api_latency_histogram_final.png", dpi=180)
        plt.show()
        plt.close(fig)
    except Exception as exc:
        print(f"Latency plot skipped: {exc}")

    print(json.dumps(summary, indent=2))
    return detail, summary


def correctness_check(base_url: str = "http://127.0.0.1:8001", n_cases: int = 12):
    from bonus import bonus_service as offline
    pre = pd.read_csv(PROJECT_ROOT / "phase1" / "outputs" / "pre_match_features_final.csv")
    mids = pre.loc[pre["season"].eq("2020/2021"), "match_id"].astype(int).tolist()
    minutes = [0, 15, 30, 45, 60, 75, 90]
    rows = []
    for i in range(n_cases):
        mid = mids[i % len(mids)]
        minute = minutes[i % len(minutes)]
        direct = offline._predict_cached(mid, minute)
        resp = requests.post(f"{base_url}/predict", json={"match_id": mid, "snapshot_minute": minute}, timeout=5)
        resp.raise_for_status()
        api = resp.json()
        prob_diff = max(abs(float(api["probabilities"][c]) - float(direct["probabilities"][c])) for c in ["H", "D", "A"])
        margin_diff = abs(float(api["expected_margin"]) - float(direct["expected_margin"]))
        rows.append({
            "match_id": mid, "snapshot_minute": minute,
            "max_probability_abs_diff": prob_diff,
            "margin_abs_diff": margin_diff,
            "pass": prob_diff < 1e-12 and margin_diff < 1e-12,
        })
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "api_correctness_final.csv", index=False)
    summary = {
        "n_cases": int(len(out)),
        "all_pass": bool(out["pass"].all()),
        "max_probability_abs_diff": float(out["max_probability_abs_diff"].max()),
        "max_margin_abs_diff": float(out["margin_abs_diff"].max()),
    }
    (OUTPUT_DIR / "api_correctness_summary_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return out, summary


if __name__ == "__main__":
    benchmark()
