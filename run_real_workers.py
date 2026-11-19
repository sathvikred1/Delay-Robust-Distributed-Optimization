"""CLI entry point. Spawn protection is required on macOS, Linux, and Windows."""
import os
# Bound native libraries before numpy/scipy imports in parent AND spawned children.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from pathlib import Path
from real_workers.experiments import run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--quick", action="store_true", help="36 short Breast Cancer demonstration runs")
    modes.add_argument("--full", action="store_true", help="216 Forest CoverType runs, seeds 11/22/33")
    parser.add_argument("--dataset", choices=("breast_cancer", "covtype"))
    parser.add_argument("--output", type=Path, help="New output directory; existing measurements are never overwritten")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--time-budget", type=float, help="Override seconds/run (recorded in manifest)")
    parser.add_argument("--max-samples", type=int, help="Explicit reproducible stratified subset, recorded in manifest")
    args = parser.parse_args()
    mode = "quick" if args.quick else "full"
    output = args.output or Path("results/real_workers" + ("/quick" if args.quick else ""))
    try:
        run_suite(mode, output, args.dataset, args.cache_dir, args.time_budget, args.max_samples)
    except (ValueError, FileExistsError, RuntimeError, TimeoutError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    print(f"Measured results saved to {output}")


if __name__ == "__main__":
    main()
