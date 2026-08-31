import time
import numpy as np
import pandas as pd
import requests


def benchmark(n_requests=100, match_id=100000, snapshot_minute=30):
    url = "http://localhost:8001/predict"
    payload = {"match_id": match_id, "snapshot_minute": snapshot_minute}
    latencies = []

    for _ in range(n_requests):
        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
        except Exception as e:
            print(f"Request failed: {e}")
            continue
        latencies.append((time.perf_counter() - start) * 1000)

    if not latencies:
        print("No successful requests.")
        return

    df = pd.DataFrame({"latency_ms": latencies})
    print(f"Latency (ms) over {len(latencies)} requests:")
    print(df.describe(percentiles=[0.50, 0.95, 0.99]))

    p95 = np.percentile(latencies, 95)
    if p95 < 200:
        print("p95 latency below 200ms -- bonus requirement met.")
    else:
        print(f"p95 latency = {p95:.2f}ms -- exceeds 200ms; consider optimising.")


if __name__ == "__main__":
    benchmark()