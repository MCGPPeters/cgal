"""Run all CGAL experiments.

Master script that runs all four experiments and collects results.
"""

import sys
from pathlib import Path
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_experiment(script_name: str):
    """Run a single experiment script."""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print('=' * 60)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Exit code: {result.returncode}")

    return result.returncode == 0


if __name__ == "__main__":
    print("=" * 60)
    print("CGAL Experiment Suite")
    print("=" * 60)
    print("\nRunning all 4 experiments...")

    experiments = [
        "baseline_regression.py",
        "continual_learning.py",
        "noise_robustness.py",
        "few_shot_learning.py"
    ]

    results = {}
    overall_start = time.time()

    for exp in experiments:
        success = run_experiment(exp)
        results[exp] = "SUCCESS" if success else "FAILED"

    overall_elapsed = time.time() - overall_start

    # Summary
    print("\n" + "=" * 60)
    print("Experiment Suite Summary")
    print("=" * 60)

    for exp, status in results.items():
        status_marker = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_marker} {exp}: {status}")

    print(f"\nTotal time: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"\nResults saved to: {Path(__file__).parent / 'results'}")
