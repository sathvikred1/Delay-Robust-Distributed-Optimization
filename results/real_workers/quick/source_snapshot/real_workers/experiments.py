"""Prespecified grids, streaming CSV output, provenance, and descriptive summaries."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .config import WorkerConfig
from .data import load_dataset
from .engine import run


def configuration_grid(mode, time_budget=None):
    variants = [("sync", None), ("async", None), ("filtered", 2),
                ("filtered", 5), ("filtered", 10), ("minibatch", 5)]
    configs = [WorkerConfig(method, workers=n, scenario=scenario, threshold=threshold,
                           seed=seed, time_budget=time_budget or (0.8 if mode == "quick" else 3.0),
                           sample_batch_size=128 if mode == "quick" else 1024,
                           evaluation_interval=0.2 if mode == "quick" else 0.5)
               for n in ([1, 4] if mode == "quick" else [1, 2, 4, 8])
               for scenario in ("natural", "moderate", "severe")
               for method, threshold in variants
               for seed in ([11] if mode == "quick" else [11, 22, 33])]
    random.Random(2028).shuffle(configs)  # Prespecified order reduces systematic thermal/order bias.
    return configs


def hardware_info():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "threadpoolctl", "pytest", "streamlit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    info = dict(platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                python=sys.version, packages=packages, multiprocessing_start_method="spawn",
                blas_threads_per_process=1,
                clock="time.perf_counter; monotonic, shared clock across local processes")
    if platform.system() == "Darwin":
        for key, name in (("cpu_model", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
            value = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
            info[key] = value.stdout.strip() if value.returncode == 0 else "unavailable"
    return info


def aggregate(raw):
    keys = ["dataset", "method", "method_label", "workers", "scenario", "threshold", "effective_batch_size"]
    metrics = [c for c in ("training_loss", "validation_loss", "validation_accuracy", "test_loss", "test_accuracy",
                           "wall_clock_seconds", "end_to_end_seconds", "startup_seconds", "optimizer_updates",
                           "gradients_computed", "gradients_computed_in_budget", "gradients_accepted",
                           "gradients_discarded", "gradients_stale", "gradients_after_budget",
                           "mean_staleness", "max_staleness", "p95_staleness", "worker_throughput",
                           "training_examples_processed", "worker_compute_seconds", "injected_delay_seconds",
                           "evaluation_seconds", "server_coordination_seconds", "delivery_latency_seconds") if c in raw]
    groups = raw.groupby(keys, dropna=False, sort=False)
    result = groups[metrics].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result["seed_count"] = groups.size()
    return result.reset_index()


def markdown_table(frame):
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def summarize(raw, manifest):
    batch_size = manifest["configurations"][0]["sample_batch_size"]
    rows = []
    for _, row in aggregate(raw).iterrows():
        rows.append({"Method": row.method_label, "Workers": row.workers, "Delay": row.scenario,
                     "Train loss": f"{row.training_loss_mean:.4f}",
                     "Validation accuracy": f"{row.validation_accuracy_mean:.3f}",
                     "Gradients/s": f"{row.worker_throughput_mean:.1f}",
                     "Mean / max staleness": f"{row.mean_staleness_mean:.2f} / {row.max_staleness_mean:.1f}"})
    focus = raw[(raw.workers == raw.workers.max()) & (raw.scenario == "severe")]
    focus = focus.groupby("method_label", sort=False).agg(loss=("training_loss", "mean"),
        accuracy=("validation_accuracy", "mean"), throughput=("worker_throughput", "mean"))
    findings = [f"- {name}: mean final training loss {r.loss:.4f}, validation accuracy "
                f"{r.accuracy:.3f}, {r.throughput:.1f} received gradients/s."
                for name, r in focus.iterrows()]
    return (f"# Measured real-worker experiment summary\n\n"
            f"Mode: **{manifest['mode']}**. Dataset: **{manifest['dataset']['name']}**. "
            f"Completed {len(raw)} / {len(manifest['configurations'])} prespecified runs.\n\n"
            f"Single-machine multiprocessing; no networked cluster was tested. "
            f"Data fingerprint: `{manifest['dataset']['sha256']}`.\n\n"
            f"Training budget: {manifest['configurations'][0]['time_budget']} seconds/run; "
            f"startup and shutdown are measured separately. All models start at zero. "
            f"Initial learning rate 0.05, sample batch {batch_size}, L2=0.001; no outcome-based tuning.\n\n"
            "## Observations from the largest worker count under severe delay\n\n" + "\n".join(findings) +
            "\n\nThese are descriptive measurements, not significance or universal speedup claims. "
            "A single quick seed has undefined sample SD. Full mode uses three seeds; "
            "OS scheduling and timing are not bitwise reproducible.\n\n"
            "## Fairness and timing\n\n"
            f"Synchronous updates use workers × {batch_size} examples; async and filtered updates use {batch_size}; "
            f"mini-batch updates combine 4 × {batch_size}. Sampling is with replacement from the same "
            "training set using independent reproducible per-worker streams. The arrival order differs. "
            "Compare loss against seconds, updates, gradients, and examples rather than update counts alone.\n\n"
            "Staleness is the number of applied server updates since a task snapshot. "
            "Only in-budget arrivals enter mean, maximum, and p95 staleness and throughput. "
            "Gradients returned after stopping are drained, recorded as after_budget, and discarded. "
            "Accepted partial groups remain pending and are never applied for free. "
            "History counts mean results received by the server at that snapshot. Raw final counts also include draining.\n\n"
            "Worker computation measures the gradient kernel including batch materialization, excluding injected sleep. "
            "Injected delays are actual worker sleeps: 5 ms moderate, 25 ms severe for the last "
            "max(1, floor(workers/4)) workers; natural adds no sleep. "
            "Communication/coordination is reported as dispatch cost, dispatch-to-start latency, "
            "finish-to-receipt latency, and parent coordination time. These overlap with computation; "
            "they are not additive CPU accounting or pure network cost. Queue wait also includes "
            "waiting for computation; it is not labeled communication overhead. Parent evaluation "
            "runs inside the training clock while workers continue. Final evaluation and draining are outside it.\n\n"
            "## Limitations\n\n"
            "Short local CPU experiments emphasize IPC and scheduling; additional workers can reduce throughput. "
            "Fixed learning rates and unequal effective batches limit optimizer comparisons. "
            "CoverType uses a declared binary one-vs-rest target; it is not seven-class classification. "
            "Random row splits can be optimistic for spatially correlated forest data. "
            "Held-out test scores are evaluated after training, never for preprocessing, stopping, or tuning. "
            "Repeated inspection of them must not guide subsequent tuning. "
            "Historical simulation uses synthetic data and simulated time, so the side-by-side figure "
            "is qualitative and cannot establish a real-world speedup.\n\n"
            "## All configurations (means; sample SD is in aggregate_results.csv)\n\n" +
            markdown_table(pd.DataFrame(rows)) + "\n")


def run_suite(mode, output, dataset=None, cache_dir="data/cache", time_budget=None, max_samples=None):
    output = Path(output)
    if (output / "experiment_config.json").exists():
        raise FileExistsError(f"Results already exist in {output}; choose --output to preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset or ("breast_cancer" if mode == "quick" else "covtype"), cache_dir, max_samples)
    configs = configuration_grid(mode, time_budget)
    root = Path(__file__).resolve().parents[1]
    paths = [*root.glob("real_workers/*.py"), root / "run_real_workers.py", root / "requirements.txt"]
    manifest = dict(mode=mode, started_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                    dataset=data.metadata, configurations=[asdict(c) for c in configs],
                    run_order_seed=2028, completed_runs=0,
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    reproducibility="Fixed data and task RNG; OS scheduling/arrival order/wall time are nondeterministic.")
    config_path = output / "experiment_config.json"
    config_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "hardware_info.json").write_text(json.dumps(hardware_info(), indent=2) + "\n")
    summaries, models = [], {}
    suite_start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for index, config in enumerate(configs, 1):
                result = run(data.problem, config)
                test_logits = data.x_test @ result.final_parameters
                test_loss = float(np.mean(np.logaddexp(0, test_logits) - data.y_test * test_logits) +
                    0.5 * data.problem.l2 * (result.final_parameters[:-1] @ result.final_parameters[:-1]))
                summary = dict(result.summary, dataset=data.metadata["name"],
                               test_loss=test_loss, test_accuracy=float(np.mean((test_logits >= 0) == data.y_test)))
                summaries.append(summary)
                models[config.run_id] = result.final_parameters
                for filename, frame in (("raw_results.csv", pd.DataFrame([summary])),
                                        ("training_history.csv", result.history),
                                        ("gradient_events.csv", result.events),
                                        ("worker_timing.csv", result.worker_timing)):
                    frame = frame.copy()
                    if filename == "gradient_events.csv":
                        frame = frame.drop(columns=["gradient_sha256"])
                    keys = ("run_id",) if filename == "gradient_events.csv" else (
                        "run_id", "method", "method_label", "workers", "scenario", "seed", "threshold")
                    for key in keys:
                        frame[key] = getattr(config, key)
                    if filename != "gradient_events.csv":
                        frame["dataset"] = data.metadata["name"]
                    frame.to_csv(output / filename, mode="w" if index == 1 else "a", header=index == 1, index=False)
                aggregate(pd.DataFrame(summaries)).to_csv(output / "aggregate_results.csv", index=False)
                manifest["completed_runs"] = index
                config_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"[{index}/{len(configs)}] {config.run_id}: loss={summary['training_loss']:.4f}, "
                      f"val_acc={summary['validation_accuracy']:.3f}, gradients={summary['gradients_computed']}, "
                      f"suite={time.perf_counter() - suite_start:.1f}s", flush=True)
    except BaseException as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    else:
        manifest["complete"] = True
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - suite_start
        config_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if summaries:
            raw = pd.DataFrame(summaries)
            (output / "experiment_summary.md").write_text(summarize(raw, manifest))
            np.savez_compressed(output / "final_models.npz", **models)
    return pd.DataFrame(summaries)
