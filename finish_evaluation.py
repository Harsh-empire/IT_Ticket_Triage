import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_step(desc, script):
    print(f"\n{'='*60}")
    print(f"STEP: {desc} ({script})")
    print(f"{'='*60}")
    cmd = [sys.executable, str(ROOT / script)]
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"FAILED: {script} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"SUCCESS: {script} completed.")

def main():
    print("Beginning automated post-BERT evaluation pipeline...")
    run_step("Train and evaluate Hybrid Transformer + SBERT Retrieval", "train_hybrid.py")
    run_step("Generate visual comparison plots", "generate_visualizations.py")
    run_step("Compile unified model comparison table", "make_comparison.py")
    print("\nALL POST-EVALUATION STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
