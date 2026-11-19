# demonstration:

Open a terminal in Delay-Robust-Distributed-Optimization, activate `.venv`, and have
the dashboard and measured figures ready. Use the saved full results for the talk;
a complete experiment grid is not a live-demo requirement.

## State the question and scope

“My project studies how worker delays and stale gradients affect SGD. Week 9 used
an event-driven simulation. I extended it with separate Python worker processes and
real datasets. This is a single-machine parameter server, not a multi-machine cluster.”

Show `archive/week9/week9_summary.md` beside `reports/real_worker_experiment_report.md`.
The former remains a simulation report. Explain that real processes expose queue,
scheduling, evaluation, and CPU-contention costs missing from simulated time.
## The implementation

Open `real_workers/engine.py`. Point to `_worker`, `WorkerPool.submit`, and `consume`
inside `run`. Each spawned worker has a different PID. Training arrays are shared
read-only; the server sends a copied model snapshot and model version through a queue.
The worker computes one gradient from that snapshot and returns it. The server applies
the returned vector without recomputing it at the newer model.

Open `worker_timing.csv` or the dashboard table to show the actual worker PIDs and
clean exits. Show the `snapshot_version`, `arrival_version`, `staleness`, and `reason`
columns in the event archive or restored CSV. Explain staleness with a simple example:
a task starts at version 5 and arrives at version 9, so staleness is 4.

## data and algorithm differences

In the dashboard, show training/validation/test counts and the declared CoverType
binary target: Lodgepole Pine versus the other six types. The scaler is fitted only
on training data. Breast Cancer is the small demonstration dataset.

Compare the four algorithm policies. Synchronous waits for every process; vanilla
async uses each arrival; filtered async rejects staleness above its threshold;
mini-batch averages four accepted gradients. Point out their different effective
batch sizes before comparing any update counts.

## Explore measured results

```bash
streamlit run dashboard/app.py
```

Select the full suite, 8 workers, severe delay, and all seeds. Compare training loss
against real time, then switch to gradients and examples. Show validation accuracy
and the worker-count/throughput comparison. Explain that more updates are not enough
to establish a better optimizer.

Show `plots/real_workers/04_gradient_staleness.png`, then
`05_threshold_comparison.png`. Read the actual values and uncertainty from the
current dashboard or CSVs rather than memorizing a preferred result.

Show `10_worker_timeline.png`. Long task bars include the deliberate worker sleep;
receipt ticks indicate when the server handled results. Natural execution adds no
artificial delay. Moderate and severe scenarios sleep only the declared slow workers.

If showing `07_simulation_vs_real.png`, explicitly say that data and time units differ.
This is a qualitative comparison of mechanisms, not a real-world speedup calculation.

## verification and reproducibility

```bash
python -m pytest -q
python scripts/real_worker_artifacts.py --results results/real_workers --verify
```

The tests cover numerical gradients, immutable snapshots, process isolation,
synchronous waiting, asynchronous updates, filtering, batching, result accounting,
data preparation, deadline crossing, timeouts, worker exceptions, and shutdown.
The independent artifact audit reconstructs counts from every returned task and
checks snapshots, timing, aggregates, source hashes, and archive checksums.

The data splits, RNG streams, configuration grid, and initial weights are reproducible.
OS scheduling and asynchronous arrival order are not bitwise reproducible.

## limitations and next work

Use the report's observations, including outcomes that contradict expectations.
Mention short runs, one machine, unequal effective batches, three seeds, and
CoverType's binary target and spatial dependence. Next steps are longer budgets,
more seeds, matched batches, spatial splits, and physical multi-machine experiments.

Optional fresh live run, if there is extra time:

```bash
python run_real_workers.py --quick --output results/real_workers/live_demo
```

This runs the full quick grid, so allow more than a minute on the recorded machine.
Use a new output directory each time; existing measurements are protected.

## Before class

- Check the dashboard loads and the full manifest says `complete: true`.
- Keep the scientific PNGs open as an offline fallback.
- Use the saved CSVs; downloading CoverType during class is unnecessary.
- To inspect the uncompressed event log after cloning, run
  `python scripts/real_worker_artifacts.py --results results/real_workers --restore-events`.
- Do not claim that the simplified methods reproduce the papers' optimality guarantees.
