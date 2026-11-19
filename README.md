# Delay-Robust Distributed Optimization

CSCI 8000 — Optimization in AI, master's research project.

This project studies how slow workers and outdated gradients affect SGD. It now
contains both the original Week 9 event-driven simulation and a separate, measured
Python multiprocessing implementation. A parameter server owns the model and
independent worker processes compute gradients from immutable model snapshots.

**Scope:** the real-worker implementation runs on one physical machine. It measures
local CPU contention, process scheduling, shared-memory access, and queue overhead.
It does not claim to be a networked multi-machine distributed system.

Repository: [sathvikred1/Delay-Robust-Distributed-Optimization](https://github.com/sathvikred1/Delay-Robust-Distributed-Optimization).
Development branch: `real-worker-experiments`. The original simulation artifacts
remain in `results/`, `plots/`, and `quick/`; new artifacts live below `real_workers/`.
The complete original Week 9 README is retained at the end of this document.

## Research questions

- How does waiting for the slowest worker affect progress under a real time budget?
- How do additional processes affect throughput, convergence, and gradient age?
- When does discarding stale work help, and how much computation does it waste?
- How does fixed asynchronous mini-batching change the effective batch size?
- Which conclusions from simulated timing remain useful with real process overhead?

## Installation

Use Python 3.10 or newer in an isolated environment. The recorded run uses the
Python and package versions in `results/real_workers/hardware_info.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements-real-workers-lock.txt` records the exact installed packages for the
new experiment. `requirements-lock.txt` is retained for the historical simulation.
For strict package replay, install the new lock file in a compatible Python environment.
A first CoverType run downloads the real data; Breast Cancer is bundled with scikit-learn.
No synthetic fallback is used if downloading fails. Dataset caches are not committed.

## Algorithms and work accounting

All methods start at the zero vector, use binary logistic loss, an unregularized
intercept, L2 regularization 0.001, and a common learning rate 0.05.
Worker batches are sampled with replacement using independent streams seeded by
experiment seed and worker ID. The same worker's task sequence is paired across
methods; asynchronous arrival order is determined by the operating system.

| Method | Server behavior | Examples per applied update |
| --- | --- | --- |
| Synchronous SGD | Wait for every worker, then average the round | workers × batch size |
| Vanilla async SGD | Apply the returned snapshot gradient immediately | batch size |
| Delay-filtered async | Apply only when staleness ≤ 2, 5, or 10 | batch size |
| Simplified async mini-batch | Average four accepted gradients, threshold 5 | 4 × batch size |

Staleness = current applied-update count − snapshot's update count. Rejections do
not increment the model version. The server never recomputes a returned gradient.
Worker results carry snapshot fingerprints so the version-to-model mapping can be audited.
The mini-batch implementation uses fixed groups; it does not reproduce the full
adaptive algorithm or theoretical parameter choices in the research papers.

At stopping, outstanding tasks are drained without updating the model. They are
recorded as `after_budget`. Incomplete accepted groups remain `gradients_pending`;
they are not flushed at zero cost. The invariants are:

```text
computed = accepted + discarded
accepted = used + pending
started = computed after draining
examples processed = computed × worker batch size
updates × effective batch size = used gradients × worker batch size
```

History logs counts of gradient results received before each model evaluation.
Raw final counts also include the drained work. Throughput and staleness summaries
use only in-budget arrivals. A discarded gradient still consumed computation.

## Real datasets and leakage prevention

| Dataset | Use | Original shape | Binary target |
| --- | --- | --- | --- |
| Breast Cancer Wisconsin Diagnostic | Quick demonstration and data tests | 569 rows, 30 features | benign = 1; malignant = 0 |
| Forest CoverType | Full main grid | 581,012 rows, 54 features | type 2, Lodgepole Pine = 1; all others = 0 |

Sources: [Breast Cancer loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
and [CoverType loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html).
CoverType originally has seven classes; this experiment explicitly uses one-versus-rest
binary logistic regression, not seven-class classification.

Both datasets use a fixed stratified 70%/15%/15% train/validation/test split with seeds
2027 and 2028. A `StandardScaler` is fitted on training rows only and then applied to
validation and test rows. A constant intercept column is appended after scaling.
Data and split fingerprints, scaler parameters, class balance, and exact row counts
are saved in `experiment_config.json`. Test scores are computed only after training;
they are not used to fit preprocessing, choose hyperparameters, or stop training.

## Reproduce experiments

```bash
python run_real_workers.py --quick
python run_real_workers.py --full
python generate_real_worker_plots.py
streamlit run dashboard/app.py
```

| Mode | Data | Workers | Seeds | Time/run | Worker batch | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| Quick | Breast Cancer | 1, 4 | 11 | 0.8 seconds | 128 | 36 |
| Full | Complete CoverType | 1, 2, 4, 8 | 11, 22, 33 | 3 seconds | 1,024 | 216 |

Both modes include natural execution, moderate delays, severe delays, and all six
algorithm variants. In moderate/severe scenarios the last `max(1, floor(workers/4))`
workers actually sleep for 5/25 ms after computing each snapshot gradient. For one
worker that worker is slow; for two workers one is slow. Natural execution adds no
sleep. The slow-worker fraction therefore varies with worker count at small counts.
Native BLAS threads are capped at one per process to avoid nested oversubscription.
Runs execute in a fixed shuffled order (seed 2028); configurations do not execute
concurrently. Runtime depends on hardware, scheduling, evaluation, and startup costs.

Quick results go to `results/real_workers/quick/`; full results go to
`results/real_workers/`. Existing result directories are protected against overwriting. Repeating the default
quick/full commands automatically creates a timestamped directory under
`results/real_workers/runs/`; the dashboard lists these new runs as well.
An explicit `--output` must name a fresh directory. For example:

```bash
python run_real_workers.py --quick --output results/real_workers/demo_run
python run_real_workers.py --full --time-budget 10 --output results/real_workers/longer_run
python run_real_workers.py --full --max-samples 50000 --output results/real_workers/subset_run
python generate_real_worker_plots.py --results results/real_workers/quick --output plots/real_workers/quick
```

Longer budgets and stratified subsets are explicit overrides recorded in the manifest;
results must not be labeled as the default full experiment when those settings differ.
The dashboard offers the committed default suites and timestamped reruns. For any
other output directory, use its CSVs and the plotting command's `--results` argument.

## Measured artifacts

| File | Contents |
| --- | --- |
| `raw_results.csv` | One row per run, losses, accuracy, counts, times, effective batches, held-out scores |
| `aggregate_results.csv` | Mean and sample SD across seeds at fixed configuration |
| `gradient_events.csv` | Every returned worker task, model versions, snapshot fingerprint, timing, acceptance |
| `training_history.csv` | Model evaluations and work counts over actual training time |
| `worker_timing.csv` | Per-process PID, compute/sleep/latency totals, throughput, clean exit |
| `experiment_config.json` | Exact grid, data fingerprints, seeds, source hashes, completion status |
| `hardware_info.json` | OS, CPU, memory, Python, packages, BLAS and process settings |
| `experiment_summary.md` | Automatically derived observations and limitations |
| `final_models.npz` | Final parameter vector for every run |
| `artifact_verification.json` | Independent cross-file accounting and checksum audit |
| `model_verification.json` | Rebuilt dataset fingerprint and recomputed scores for every final model |

Full event CSVs can exceed GitHub's individual-file limit. **Every measured event is
retained:** the local CSV is saved as requested and exact compressed CSV parts are
versioned under `gradient_event_parts/`. These contain no sampled or fabricated rows.
The index records row counts and SHA256 checksums. Plots read either the local CSV or
the versioned parts. Restore the byte-identical CSV after cloning with:

```bash
python scripts/real_worker_artifacts.py --results results/real_workers --restore-events
python scripts/real_worker_artifacts.py --results results/real_workers/quick --restore-events
```

Archive and independently verify a new run with:

```bash
python scripts/real_worker_artifacts.py --results results/real_workers --archive --verify
python scripts/verify_real_worker_models.py --results results/real_workers
```

## Results and figures

<!-- REAL_WORKER_FINDINGS_START -->
On the recorded Apple M2, vanilla async's natural-execution throughput was 4,046, 7,244, 10,386, 10,322 gradients/s for 1, 2, 4, and 8 workers. Throughput flattened between 4 and 8 workers.

With 8 workers and severe delays, mean final training loss was 0.57183 for synchronous SGD and 0.51596 for vanilla async. Effective batches and total work differ, so these are not batch-matched optimizer rankings.

Vanilla async's mean maximum staleness increased from 24.3 updates under natural execution to 391.3 under severe delay, while mean staleness stayed near seven. These are descriptive three-seed results from short local runs.
<!-- REAL_WORKER_FINDINGS_END -->

Read the measured [full summary](results/real_workers/experiment_summary.md),
[quick summary](results/real_workers/quick/experiment_summary.md), and
[research report](reports/real_worker_experiment_report.md). They distinguish the
real experiments from the original [Week 9 simulation report](results/week9_summary.md).
No earlier report is rewritten to imply that real-worker experiments had already happened.

`plots/real_workers/` contains readable PNG and vector PDF versions of:

1. Training loss versus measured seconds.
2. Validation accuracy versus measured seconds.
3. Worker count versus gradient throughput, separated by delay scenario.
4. Mean and maximum gradient staleness.
5. Filtering threshold versus final loss and stale-work rejection.
6. Slow-worker effects on loss and throughput.
7. Historical simulation and real-worker curves in separately labeled panels.
8. Loss versus computed gradients received by the server.
9. Loss versus training examples processed, including repetitions.
10. A measured per-worker execution and server-receipt timeline.
11. Loss versus applied optimizer updates.

Source snapshots preserve the exact experiment files matching each run's hashes,
even if CLI or reporting tools are later improved. The audit reports current-code
differences separately from the verified run-source snapshot.

Seed curves are linearly interpolated only over their shared progress range. Bands
are sample standard deviations, not confidence intervals. Quick mode has one seed,
so its sample SD is undefined and no uncertainty band is drawn. The simulation panel
uses synthetic data and simulated time units: it is a qualitative comparison, not a
calibration or speedup measurement against the real-data experiments.

![Measured worker scaling](plots/real_workers/03_worker_scaling.png)

## Dashboard and demonstration

`streamlit run dashboard/app.py` opens the interactive results explorer. It reads CSVs
directly and derives all displayed observations. Filter by dataset suite, algorithm,
worker count, delay scenario, and seed, then compare time, updates, gradients, and
examples. It also shows staleness, throughput, effective batch sizes, hardware details,
and downloadable result rows.

Review [verification evidence](reports/verification_summary.md) and the
[changed-file manifest](reports/changed_files.md) before pushing.
Start with the [simple explanation guide](reports/professor_explanation_guide.md) if you are new to the experiment.
Use [the professor demo](reports/professor_demo.md) for a 5–7 minute walkthrough,
and [the semester progress plan](reports/semester_progress.md) to connect this work
to Weeks 10–15.

## Interpretation and limitations

More updates do not establish better optimization. Methods have different effective
batches and may do different amounts of work within the same clock budget. This is
a systems-and-optimization study, not a batch-matched ranking of optimizers.

Initial evaluation and process startup are outside the training budget and reported
separately; periodic full training/validation evaluation is inside it while workers
continue. Final evaluation, draining, and shutdown are outside it. End-to-end time
includes all these phases. Held-out test scoring, dataset loading, and CSV writing are suite costs, not
per-run training costs. A running evaluation can slightly overrun the nominal deadline;
actual elapsed time is recorded and no later arrival updates the model.

Gradient compute time excludes injected sleep. Dispatch, dispatch-to-start latency,
result delivery latency, server coordination, and queue wait are separate measurements.
They overlap with work in other processes and cannot be summed as a pure communication
cost. Queue waiting also includes computation. No network latency was measured.

Short logistic-regression kernels can be dominated by IPC and server evaluation.
Local scheduling, background machine activity, and CPU contention make arrival order,
wall time, and final asynchronous parameters nondeterministic even with fixed RNG seeds.
Three seeds support descriptive mean/SD comparisons, not statistical-significance claims.
Random row splits of spatial forest observations may overstate geographic generalization.
Fixed learning rates, unequal batches, a binary target, one machine, and fixed injected
sleeps limit generality. Longer runs, more seeds, matched batches, spatial holdouts,
and a networked multi-machine implementation are future work.

## Research papers

- Amit Attia, Ofir Gaash, Tomer Koren. [Faster Stochastic Optimization with Arbitrary
  Delays via Adaptive Asynchronous Mini-Batching](https://proceedings.mlr.press/v267/attia25a.html), ICML 2025.
- Artavazd Maranjyan, Alexander Tyurin, Peter Richtárik. [Ringmaster ASGD: The First
  Asynchronous SGD with Optimal Time Complexity](https://proceedings.mlr.press/v267/maranjyan25b.html), ICML 2025.

These papers motivate stale-gradient rejection and asynchronous batching. The project
implements simplified educational policies, not the papers' full theoretical algorithms.

---

<details>
<summary>Original Week 9 simulation README (preserved historical results)</summary>

# Asynchrony / Delay-Robust Optimization

This project studies how worker delays and stale gradients affect distributed stochastic optimization. We use an event-driven simulation to compare synchronous SGD with several asynchronous optimization approaches under different levels of worker delay.

The implementation is mainly inspired by two recent papers:

1. **Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching** — Attia, Gaash & Koren, ICML 2025
2. **Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity** — Maranjyan, Tyurin & Richtárik, ICML 2025

This project is a simplified educational implementation for studying the main ideas. It is not intended to be an exact reproduction of either paper.

---

## Methods Implemented

We compare four main optimization approaches.

### 1. Synchronous SGD

All workers compute gradients, but the system waits for every worker before performing the next model update.

This means that a slow worker can delay the entire optimization process.

### 2. Vanilla Asynchronous SGD

Workers operate independently. When a worker finishes computing a gradient, that gradient is immediately applied to the current model.

The gradient is computed using the model snapshot available when the worker started its task. Therefore, slow workers may return gradients based on older versions of the model.

### 3. Delay-Filtered Asynchronous SGD

This method is similar to vanilla Async-SGD, but very stale gradients can be rejected.

We experiment with three staleness thresholds:

- tau = 2
- tau = 5
- tau = 10

This allows us to study the trade-off between using more gradients and rejecting outdated information.

### 4. Simplified Asynchronous Mini-Batch SGD

This method collects accepted asynchronous gradients and averages them in groups of four before performing a model update.

This is inspired by adaptive asynchronous mini-batching, but our implementation uses a fixed batch size and fixed staleness threshold.

---

# Experimental Setup

The optimization problem is binary logistic regression on a synthetic classification dataset.

The full experiment uses:

- 6,000 total samples
- 30 features
- 4,800 training samples
- 1,200 validation samples
- L2 regularization coefficient = 0.001
- Learning rate = 0.15
- Worker gradient batch size = 32
- Worker counts = 4, 8, and 16
- Three delay conditions: low, moderate, and high heterogeneity
- Three random seeds: 11, 22, and 33
- Full simulated time horizon = 150 units

The full experiment contains **162 main configurations**, along with additional controlled straggler experiments.

The dataset and train/validation split are fixed across the methods so that the methods can be compared under the same data conditions.

---

# Running the Project

## Install Dependencies

From the project directory:

```bash
python -m pip install -r requirements.txt
```

For running the tests:

```bash
python -m pip install pytest
```

---

## Run Tests

Before running the experiments:

```bash
python -m pytest -q
```

The committed experiment artifacts report that all **12 unit tests passed**.

---

## Quick Experiment

For a smaller test run:

```bash
python main.py --quick
```

Quick mode runs:

- 54 configurations
- 1 random seed
- 40 simulated time units

The results are saved under:

```text
quick/results/
quick/plots/
```

This mode is useful for checking that the implementation works before running the complete experiment.

---

## Full Experiment

Run:

```bash
python main.py --full
```

Full mode uses:

- 162 main runs
- 3 random seeds
- 150 simulated time units

The generated results are saved in:

```text
results/
```

and figures are saved in:

```text
plots/
```

---

# Experimental Results

The following results focus on the important case of **8 workers under high delay heterogeneity**.

The values are averages across three random seeds.

| Method | Train Loss | Validation Accuracy | Updates | Mean / Max Staleness | Discard % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous SGD | 0.4002 | 0.860 | 20.0 | 0.00 / 0.0 | 0.0% |
| Vanilla Async-SGD | 0.3225 | 0.872 | 937.0 | 6.93 / 67.3 | 0.0% |
| Filtered Async, tau=2 | 0.3227 | 0.875 | 464.3 | 3.43 / 35.0 | 50.4% |
| Filtered Async, tau=5 | 0.3222 | 0.871 | 720.0 | 5.32 / 53.3 | 23.2% |
| Filtered Async, tau=10 | 0.3225 | 0.872 | 874.0 | 6.46 / 64.3 | 6.7% |
| Async Mini-Batch, B=4, tau=5 | **0.3210** | **0.876** | 223.0 | **1.65 / 16.3** | 4.6% |

---

# Main Observations

## 1. Synchronous vs. Asynchronous SGD

One of the clearest differences appears when worker speeds are highly heterogeneous.

With 8 workers and the same simulated time budget:

- Synchronous SGD completed about **20 updates**
- Vanilla Async-SGD completed about **937 updates**

Their final mean training losses were:

- Synchronous SGD: **0.4002**
- Vanilla Async-SGD: **0.3225**

The main reason is that synchronous SGD must wait for the slowest worker before continuing.

Async-SGD can continue processing gradients from faster workers instead of waiting.

However, the synchronous and asynchronous methods use different effective update batch sizes, so this result should not be interpreted as a perfectly batch-matched performance comparison.

---

## 2. Effect of the Staleness Threshold

The filtering experiments show an interesting trade-off.

For 8 workers under high heterogeneity:

| Threshold | Train Loss | Gradients Discarded |
| ---: | ---: | ---: |
| tau=2 | 0.3227 | 50.4% |
| tau=5 | **0.3222** | 23.2% |
| tau=10 | 0.3225 | 6.7% |

A very strict threshold was not automatically better.

With tau=2, approximately half of the gradients were discarded. However, this did not produce the lowest final loss.

Among the filtering thresholds tested, **tau=5 produced the lowest mean final training loss**.

This suggests that there is a trade-off between removing stale information and throwing away potentially useful gradient work.

These comparisons are descriptive results from three seeds and are not statistical-significance claims.

---

## 3. Effect of Increasing the Number of Workers

For vanilla Async-SGD under high heterogeneity:

| Workers | Train Loss | Mean Staleness | Max Staleness | Completed Gradients |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.3222 | 2.97 | 30.7 | 471.7 |
| 8 | 0.3225 | 6.93 | 67.3 | 937.0 |
| 16 | 0.3225 | 14.87 | 164.3 | 1890.0 |

Increasing the number of workers greatly increased the number of completed gradients.

However, it also increased gradient staleness.

Mean staleness increased approximately from:

```text
2.97 -> 6.93 -> 14.87
```

while maximum staleness increased from:

```text
30.7 -> 67.3 -> 164.3
```

This demonstrates an important trade-off in asynchronous optimization.

More workers can increase the amount of work completed, but gradients can become increasingly outdated before they reach the global model.

In this experiment, increasing the worker count also did not continuously improve the final training loss.

---

# Unexpected Result: Mean vs. Tail Staleness

One interesting result was that increasing worker-delay heterogeneity did **not** noticeably increase the average completed-gradient staleness.

For vanilla Async-SGD with 8 workers:

```text
Low heterogeneity mean staleness  = 6.98
High heterogeneity mean staleness = 6.93
```

At first, we expected the mean staleness to increase.

However, the tail of the staleness distribution changed significantly.

Mean maximum staleness increased from:

```text
11.7 -> 67.3
```

and mean p95 staleness increased from:

```text
9.00 -> 16.53
```

This indicates that worker heterogeneity creates a smaller number of extremely stale gradients even when the overall mean stays relatively stable.

In a closed system with continuously active workers, average update-based staleness can remain close to the number of other active workers. Because of this, statistics such as maximum and 95th-percentile staleness can provide more useful information about severe worker delays than the mean alone.

The experiment keeps this unexpected result instead of changing the simulation to force the originally expected trend.

---

# Simplified Async Mini-Batching

The simplified asynchronous mini-batch method also produced interesting results.

With 8 workers under high heterogeneity:

| Method | Train Loss | Updates | Mean Arrival Staleness |
| --- | ---: | ---: | ---: |
| Vanilla Async-SGD | 0.3225 | 937.0 | 6.93 |
| Async Mini-Batch | **0.3210** | 223.0 | **1.65** |

The mini-batch method performs fewer model updates because it combines four accepted gradients before updating the model.

Therefore, its update count and update-based staleness should not be compared directly with vanilla Async-SGD without considering this difference.

Our implementation is intentionally simplified and does not reproduce the complete adaptive mini-batching algorithm from the original paper.

---

# Verification and Sanity Checks

The generated full-run artifacts contain several checks to make sure that the results come from the actual simulation.

The recorded verification includes:

- **162 main experimental runs**
- **54 aggregate configurations**
- **174 runs checked** including diagnostic experiments
- **12 unit tests passed**
- Algorithm invariants passed for all checked runs
- Exact reproducibility checks passed
- Training loss decreased from its initial value in **162/162 main runs**
- Filtering was active in **76/81 applicable runs**
- Source hashes matched the experiment manifest
- Summary statistics matched independent recalculation
- Quick-run events matched prefixes of the corresponding full runs
- 12 PNG figures generated
- 12 vector PDF figures generated

Detailed verification information can be found in:

```text
results/sanity_checks.json
results/artifact_verification.json
```

---

# Important Result Files

The most useful generated files are:

### `results/week9_summary.md`

Automatically generated readable summary of the experimental setup, results, observations, sanity checks, limitations, and future experiments.

### `results/summary_results.csv`

Individual experimental results for each random seed.

### `results/aggregate_results.csv`

Mean and sample standard deviation across seeds for each experimental configuration.

### `results/raw_results.csv`

Complete optimization history recorded during the experiments.

### `results/gradient_events.csv.gz`

Individual worker gradient-arrival events, including gradient staleness and whether the gradient was accepted.

### `results/paired_comparisons.csv`

Paired comparisons between selected methods.

### `results/empirical_checks.csv`

Checks whether expected experimental trends were actually observed.

### `results/sanity_checks.json`

Algorithm invariants and reproducibility results.

### `results/run_manifest.json`

Experiment configuration, package versions, dataset fingerprint, seeds, and source-code hashes.

### `results/artifact_verification.json`

Verification of the generated experiment artifacts.

---

# Important Figures

The experiment generates several plots.

Some of the most useful figures for the project report are:

```text
plots/02_loss_vs_time.png
```

Shows optimization loss against simulated wall-clock time.

```text
plots/06_worker_scaling.png
```

Shows the effect of using 4, 8, and 16 workers.

```text
plots/07_threshold_comparison.png
```

Shows the effect of different stale-gradient filtering thresholds.

```text
plots/10_tail_staleness.png
```

Shows how heterogeneous worker delays affect the tail of the staleness distribution.

```text
plots/12_loss_vs_gradient_work.png
```

Compares optimization progress relative to gradient computation work.

---

# Limitations

There are several limitations to the current experiment.

First, this is an event-driven simulation rather than a real distributed computing cluster. Computation is performed locally while worker completion times are simulated.

The experiment therefore does not include:

- Network communication cost
- Server bottlenecks
- Hardware contention
- Worker failures
- Scheduling overhead
- Cancellation overhead

The experiments also use one synthetic convex logistic-regression problem and a fixed learning rate.

In addition, the methods do not have identical effective update batch sizes. Synchronous SGD averages gradients from all workers, vanilla Async-SGD applies one gradient at a time, and the simplified mini-batch method combines four gradients.

Therefore, these experiments should be viewed as a controlled study of **worker delays, throughput, gradient staleness, and stale-gradient filtering**, rather than proof that one method is universally better than another.

---

# Future Work

Possible next experiments include:

1. Testing multiple learning rates using a prespecified common learning-rate sweep.
2. Comparing methods under matched effective batch sizes.
3. Increasing the number of random seeds.
4. Testing additional datasets.
5. Studying non-convex optimization problems.
6. Simulating time-varying worker speeds.
7. Adding communication and server overhead.
8. Testing larger numbers of workers.
9. Implementing the complete adaptive mini-batching procedure from the research paper.
10. Implementing more of the theoretical Ringmaster ASGD parameter-selection procedure.

---

# References

**Amit Attia, Ofir Gaash, and Tomer Koren.**  
*Faster Stochastic Optimization with Arbitrary Delays via Adaptive Asynchronous Mini-Batching.*  
International Conference on Machine Learning (ICML), 2025.

**Artavazd Maranjyan, Alexander Tyurin, and Peter Richtárik.**  
*Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity.*  
International Conference on Machine Learning (ICML), 2025.

</details>
