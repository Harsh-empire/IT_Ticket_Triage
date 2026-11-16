import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    "prepare_data.py",
    "train_classical.py",
    "train_sbert_xgboost.py",
    "train_transformers.py",
    "train_hybrid.py",
    "make_comparison.py",
]


def main():
    for script in STEPS:
        print("\n" + "=" * 72)
        print(f"Running {script}")
        print("=" * 72)

        completed = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            check=False,
        )

        if completed.returncode != 0:
            raise SystemExit(
                f"\nStopped because {script} returned exit code "
                f"{completed.returncode}. Fix that error, then rerun the "
                "failed step rather than retraining models that already finished."
            )

    print("\nAll experiments completed.")
    print("Open results/model_comparison.csv for the measured comparison table.")


if __name__ == "__main__":
    main()
