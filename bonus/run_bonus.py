import argparse
import subprocess
import sys
import time
from pathlib import Path
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", action="store_true", help="Start FastAPI service only")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard only")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmark")
    parser.add_argument("--demo", action="store_true", help="Start service + dashboard together (default)")

    args = parser.parse_args()
    if not any(vars(args).values()):
        args.demo = True

    dashboard_path = Path("bonus_dashboard.py")

    if args.service:
        print("Starting FastAPI service on http://localhost:8001")
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "bonus_service:app",
             "--reload", "--host", "0.0.0.0", "--port", "8001"]
        )
    elif args.dashboard:
        print("Launching Streamlit dashboard on http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
    elif args.latency:
        from bonus_latency import benchmark
        benchmark()
    elif args.demo:
        print("Starting FastAPI service on http://localhost:8001 ...")
        service_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "bonus_service:app",
             "--host", "0.0.0.0", "--port", "8001"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        print("Waiting for service to start...")
        for _ in range(15):
            try:
                if requests.get("http://localhost:8001/health", timeout=1).status_code == 200:
                    print("Service is ready.")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            print("Service did not start in time.")
            service_process.terminate()
            sys.exit(1)

        print("Launching Streamlit dashboard on http://localhost:8501 ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        service_process.terminate()


if __name__ == "__main__":
    main()