from pathlib import Path
from phase1.final_feature_engineering import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run(root)
