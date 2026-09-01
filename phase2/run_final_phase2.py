from pathlib import Path
from phase2.final_defense_pipeline import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
