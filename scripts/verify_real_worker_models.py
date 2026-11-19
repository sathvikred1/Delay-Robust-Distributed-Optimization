"""Rebuild the real dataset split and independently score every saved final model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_workers.data import load_dataset


def verify_models(directory, cache_dir):
    directory = Path(directory)
    manifest = json.loads((directory / "experiment_config.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot verify final models for an incomplete experiment")
    description = manifest["dataset"]
    subset = description["selected_samples"] if description["selected_samples"] < description["source_samples"] else None
    errors = []
    with threadpool_limits(limits=1):
        data = load_dataset(description["name"], cache_dir, subset)
        fingerprint_matches = data.metadata["sha256"] == description["sha256"]
        if not fingerprint_matches:
            errors.append("Prepared dataset fingerprint differs from the run manifest")
        raw = pd.read_csv(directory / "raw_results.csv")
        with np.load(directory / "final_models.npz", allow_pickle=False) as models:
            if set(models.files) != set(raw.run_id):
                errors.append("Saved model IDs differ from measured run IDs")
            for row in raw.itertuples():
                parameters = models[row.run_id]
                metrics = data.problem.evaluate(parameters)
                logits = data.x_test @ parameters
                metrics["test_loss"] = float(np.mean(np.logaddexp(0, logits) - data.y_test * logits) +
                    0.5 * data.problem.l2 * np.dot(parameters[:-1], parameters[:-1]))
                metrics["test_accuracy"] = float(np.mean((logits >= 0) == data.y_test))
                for name, value in metrics.items():
                    if not np.isclose(value, getattr(row, name), rtol=1e-10, atol=1e-12):
                        errors.append(f"Saved model score differs: {row.run_id}, {name}")
    report = dict(passed=not errors, dataset_fingerprint_matches=fingerprint_matches,
                  models_checked=len(raw), errors=errors,
                  metrics=["training_loss", "validation_loss", "training_accuracy", "validation_accuracy", "test_loss", "test_accuracy"])
    (directory / "model_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise AssertionError(errors)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/real_workers")
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()
    print(verify_models(args.results, args.cache_dir))
