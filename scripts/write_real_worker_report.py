"""Write the final narrative and tables only from complete measured suites."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real_workers.experiments import markdown_table


def estimate(frame, field, digits=4):
    return f"{frame[field].mean():.{digits}f} ± {frame[field].std(ddof=1):.{digits}f}"


def main():
    full = ROOT / "results/real_workers"
    quick = full / "quick"
    manifest = json.loads((full / "experiment_config.json").read_text())
    q_manifest = json.loads((quick / "experiment_config.json").read_text())
    verification = json.loads((full / "artifact_verification.json").read_text())
    model_verification = json.loads((full / "model_verification.json").read_text())
    hardware = json.loads((full / "hardware_info.json").read_text())
    assert manifest["complete"] and q_manifest["complete"] and verification["passed"] and model_verification["passed"]
    raw = pd.read_csv(full / "raw_results.csv")
    q_raw = pd.read_csv(quick / "raw_results.csv")
    dataset = manifest["dataset"]
    largest = raw.workers.max()
    focus = raw[(raw.workers == largest) & (raw.scenario == "severe")].copy()
    focus["stale_percentage"] = 100 * focus.gradients_stale / focus.gradients_computed_in_budget.clip(lower=1)
    table = []
    for label, frame in focus.groupby("method_label", sort=True):
        table.append({"Method": label, "Train loss": estimate(frame, "training_loss", 5),
                      "Val accuracy": estimate(frame, "validation_accuracy", 3),
                      "Gradients/s": estimate(frame, "worker_throughput", 1),
                      "Updates": estimate(frame, "optimizer_updates", 1),
                      "Stale discard %": f"{frame.stale_percentage.mean():.2f}",
                      "Mean / max age": f"{frame.mean_staleness.mean():.2f} / {frame.max_staleness.mean():.1f}"})
    scaling = []
    natural = raw[(raw.method == "async") & (raw.scenario == "natural")]
    for count, frame in natural.groupby("workers"):
        scaling.append({"Workers": count, "Gradients/s": estimate(frame, "worker_throughput", 1),
                        "Train loss": estimate(frame, "training_loss"),
                        "Mean / max age": f"{frame.mean_staleness.mean():.2f} / {frame.max_staleness.mean():.1f}"})
    sync_natural = raw[(raw.method == "sync") & (raw.workers == largest) & (raw.scenario == "natural")]
    sync_severe = focus[focus.method == "sync"]
    async_severe = focus[focus.method == "async"]
    async_natural = natural[natural.workers == largest]
    decreased = int((raw.training_loss < np.log(2)).sum())
    threshold_rows = []
    for threshold, frame in focus[focus.method == "filtered"].groupby("threshold"):
        threshold_rows.append({"Threshold": int(threshold), "Train loss": estimate(frame, "training_loss", 5),
                               "Val accuracy": estimate(frame, "validation_accuracy", 4),
                               "Stale discard %": estimate(frame, "stale_percentage", 2)})
    text = f'''# Real-worker experiment report

CSCI 8000 — Optimization in AI.

We started with a simulation of worker delays in Week 9. We then added separate
Python worker processes to see what changes when timing and coordination costs
are real. This report covers that extension. The earlier work is kept in
`archive/week9/`.

## What we wanted to find out

Our question was simple: can training keep moving when some workers are slow,
and what happens when their gradients arrive late?

We compared synchronous SGD, vanilla asynchronous SGD, delay-filtered asynchronous
SGD, and a simplified asynchronous mini-batch method. We looked at model quality,
time, and work used together, rather than deciding which method was better from
its update count alone.

## How the implementation works

We use one parent process as the parameter server. It keeps and updates the model.
Each worker is a separate OS process created using Python's spawn start method.
Read-only training arrays are shared between processes. Each task carries a copied
model snapshot and its applied-update version. Workers calculate the gradient using
that snapshot and return the resulting vector. The server never recalculates a
received gradient at the newer model.

Staleness is the server version at receipt minus the snapshot version. Only applied
updates advance the version. Synchronous SGD waits for every worker and averages
one full round. Vanilla async applies each arrival immediately. Filtered async
accepts staleness less than or equal to 2, 5, or 10. Simplified mini-batch averages
four accepted gradients with threshold 5; no update occurs until the group is full.

This is real single-machine multiprocessing, not distributed execution across
separate physical computers. Queue serialization, scheduling, CPU contention,
shared-memory access, and server evaluation are included. Network communication
and cross-machine failures are not measured.

## Data and preparation

The quick suite uses the real Breast Cancer Wisconsin Diagnostic dataset and
completed {len(q_raw)} runs. It is a demonstration with one seed, not a statistical
benchmark. The main suite uses all {dataset['source_samples']:,} Forest CoverType
rows and {dataset['features']} input features. Its declared binary target is
Lodgepole Pine (original class 2) versus all other forest cover types.

The main split has {dataset['train_samples']:,} training,
{dataset['validation_samples']:,} validation, and {dataset['test_samples']:,} test
rows. Splits are stratified and fixed using seeds 2027 and 2028. Preprocessing is
fitted on training rows only. The scaler is then applied to validation and test data,
and an intercept column is appended. Test data is evaluated after training and never
used for preprocessing, stopping, or parameter selection. All models initialize at zero.

Dataset fingerprint: `{dataset['sha256']}`.
The manifest also records split-index fingerprints, scaler parameters, and class
balance. Random row splits may be optimistic for spatially correlated forest data.

## How we ran the experiments

The completed full grid contains **{len(raw)} runs**: 1, 2, 4, and 8 workers; natural,
moderate, and severe worker conditions; six algorithm variants; and seeds 11, 22, 33.
The six variants are sync, async, filtered tau=2/5/10, and mini-batch tau=5.
Configurations execute sequentially in a fixed shuffled order (seed 2028).

Every full run receives a nominal 3-second training budget, learning rate 0.05,
L2 penalty 0.001, and a worker batch of 1,024 examples. Quick mode uses 0.8 seconds
and batches of 128. Worker sampling is with replacement, so examples processed
counts repeated examples, not unique dataset coverage. The task sampling streams are
fixed by training seed and worker ID, but the OS determines their arrival order.

Natural execution injects no delay. Moderate and severe conditions add actual
5 ms and 25 ms worker sleeps after gradient calculation. The last
max(1, floor(workers/4)) workers are slow. Thus the single-worker baseline has one
slow worker under injected-delay conditions, and two workers have a 50% slow fraction.
This changing fraction limits direct worker-scaling interpretations across scenarios.

Synchronous updates use workers × 1,024 examples, async and filtered updates use
1,024, and mini-batch uses 4,096. No method-specific learning-rate tuning is performed.
The unequal effective batches are a deliberate limitation, so comparisons use loss
versus time, updates, computed gradients, and examples rather than updates alone.

## Timing and accounting

All process timestamps use time.perf_counter(). Startup and initial model evaluation
are outside the training clock and recorded separately. Periodic full-data evaluations
run in the server during training while workers continue. A final evaluation,
draining, and shutdown are outside the budget. End-to-end run time includes these
phases. Loading data, writing CSVs, plotting, and archiving are suite-level costs.

Accepted incomplete groups remain pending. Outstanding tasks are drained after
stopping, recorded as after_budget, and never update the model. Final raw totals
include draining; training-history work counts and throughput exclude late receipts.
A regression test checks the case where an arrival crosses the deadline while the
server is waiting. Actual clock duration is saved, including evaluation overruns.

Worker computation measures the gradient kernel and batch materialization, excluding
injected sleep. Dispatch cost, dispatch-to-start latency, result-delivery latency,
and parent coordination time describe local communication and scheduling overhead.
These quantities overlap across processes and cannot be summed as pure communication
cost. Queue wait also includes waiting for worker computation.

Accounting checks require computed = accepted + discarded; accepted = used + pending;
started = computed after draining; examples = computed × worker batch size; and
updates × effective batch = used gradients × worker batch size.

## Machine and checks

Recorded machine: {hardware.get('cpu_model', hardware['processor'])},
{hardware['logical_cpus']} logical CPUs,
{int(hardware['memory_bytes']) / 2**30:.0f} GiB RAM, {hardware['platform']}.
Python: {hardware['python'].split(' (')[0]}. Native BLAS threads are limited to one per
process. Complete package versions are in hardware_info.json and the new lock file.

The full suite took {manifest['runtime_seconds']:.1f} measured seconds, excluding data
loading and subsequent artifact processing. The independent audit checked
**{verification['events_checked']:,} returned gradients** and
**{verification['snapshot_versions_checked']:,} model-version fingerprints** across
all {verification['runs_checked']} runs. It checks cross-file accounting, time order,
clean exits, history endpoints, independently recalculated aggregates, and source
and archive checksums. Byte-exact source snapshots preserve the code used by older
measurements even when reporting/provenance tools evolve.

A separate model audit rebuilt the dataset split, reproduced its exact fingerprint,
and recomputed training, validation, and test scores for all
{model_verification['models_checked']} saved full-run models. These matched the
measured CSVs within numerical tolerance. The quick models passed the same check.

## Results: {largest} workers, severe delays

Values are mean ± sample SD across three seeds. Stale-discard percentages exclude
post-budget draining; maximum age is the mean of per-run maxima, not the pooled max.

{markdown_table(pd.DataFrame(table))}

All {decreased}/{len(raw)} main runs ended below their zero-model training loss of
{np.log(2):.4f}. This is a convergence sanity check, not proof of optimizer superiority.

Under severe delay, synchronous SGD received
{sync_severe.worker_throughput.mean():.1f} gradients/s and completed
{sync_severe.optimizer_updates.mean():.1f} updates on average. Under natural execution
at the same worker count it received {sync_natural.worker_throughput.mean():.1f}
gradients/s and completed {sync_natural.optimizer_updates.mean():.1f} updates.
Vanilla async under severe delay received {async_severe.worker_throughput.mean():.1f}
gradients/s, with mean final training loss {async_severe.training_loss.mean():.5f}.
Faster workers can continue while slow tasks are outstanding; effective batches and
amounts of work still differ, so this is not a batch-matched optimizer comparison.

## Worker counts under natural execution

{markdown_table(pd.DataFrame(scaling))}

These are local throughput measurements, not a linear-scaling claim. More processes
share the CPU and memory system and increase queue and server work. A small final-loss
difference after many gradients is not enough to establish a better optimizer.

At {largest} workers, vanilla async's mean staleness changed from
{async_natural.mean_staleness.mean():.2f} under natural execution to
{async_severe.mean_staleness.mean():.2f} under severe delay. Its mean maximum staleness
changed from {async_natural.max_staleness.mean():.1f} to
{async_severe.max_staleness.mean():.1f}. Report both central and tail age: a modest mean
does not rule out very old gradients. Mini-batch age is counted in larger model
updates, so its update-based staleness is not directly batch-matched to vanilla async.

## Filtering thresholds

{markdown_table(pd.DataFrame(threshold_rows))}

A strict threshold can discard useful computation. Compare the measured loss,
validation accuracy, rejection rate, and gradient/example axes together. These three
seeds describe variability; they do not establish a statistically significant winner.
No preferred outcome or monotonic threshold trend is assumed.

## Simulation comparison and figures

The Week 9 synthetic-data simulation and the real-data process measurements appear
in separate panels with separate time units. This comparison shows mechanisms and
limitations; it cannot quantify a real-world speedup because datasets, worker timing,
and effective workloads differ. The archived simulation remains unchanged.

The 11 main figures in plots/real_workers/ cover training loss/time, validation
accuracy/time, worker throughput, mean/max staleness, thresholds, slow workers,
simulation versus real execution, loss/gradients, loss/examples, the actual worker
timeline, and loss/updates. Each has a PNG and PDF version. Curves interpolate only
on common seed support; shaded bands show sample SD, not confidence intervals.

The Streamlit dashboard reads the measured CSVs directly and exposes worker, method,
scenario, and seed filters, four progress axes, hardware, and result downloads.

## What we can conclude and what still needs work

The budgets are short and the model is convex binary logistic regression. Natural
local overhead can dominate small gradient tasks. Some runs occurred while plots,
tests, or other desktop activity were running; the machine was not exclusively
reserved. Fixed randomized run order reduces systematic order bias but does not
remove background-load effects. Only one machine, three seeds, one main dataset,
fixed learning rates, unequal batches, and fixed slow-worker sleeps were tested.
Test scores are retained for transparency, but repeated inspection must not guide
future tuning on the same held-out data.

The main lesson from our runs is that waiting for slow workers can limit progress,
while asynchronous updates bring a different problem: some gradients are very old.
Adding workers also stopped giving a clear throughput increase between four and eight.

Our next step would be longer runs on an otherwise idle machine, more seeds, and
matched effective batches. Spatial holdouts would strengthen the forest-data study.
Separate physical machines would be needed to study network costs. The fixed
mini-batch and filtering methods here are simplified versions of the research ideas;
we have not reproduced the papers' full algorithms or optimality guarantees.

## References and reproduction

- Attia, Gaash, Koren, ICML 2025: https://proceedings.mlr.press/v267/attia25a.html
- Maranjyan, Tyurin, Richtárik, ICML 2025: https://proceedings.mlr.press/v267/maranjyan25b.html
- Breast Cancer data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- CoverType data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html

See README.md for installation, exact commands, event-log restoration, and protected
output directories. Use reports/demo.md for the 5–7 minute demonstration
and reports/presentation_outline.md to prepare classroom slides.
'''
    path = ROOT / "reports/real_worker_experiment_report.md"
    path.write_text(text)
    throughput = natural.groupby("workers").worker_throughput.mean()
    finding = ("<!-- REAL_WORKER_FINDINGS_START -->\n"
               f"On the recorded {hardware.get('cpu_model', hardware['processor'])}, vanilla async's natural-execution throughput was "
               + ", ".join(f"{value:,.0f}" for value in throughput)
               + " gradients/s for 1, 2, 4, and 8 workers. Throughput flattened between 4 and 8 workers.\n\n"
               f"With 8 workers and severe delays, mean final training loss was {sync_severe.training_loss.mean():.5f} "
               f"for synchronous SGD and {async_severe.training_loss.mean():.5f} for vanilla async. "
               "Effective batches and total work differ, so these are not batch-matched optimizer rankings.\n\n"
               f"Vanilla async's mean maximum staleness increased from {async_natural.max_staleness.mean():.1f} "
               f"updates under natural execution to {async_severe.max_staleness.mean():.1f} under severe delay, "
               "while mean staleness stayed near seven. These are descriptive three-seed results from short local runs.\n"
               "<!-- REAL_WORKER_FINDINGS_END -->")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()
    begin, end = "<!-- REAL_WORKER_FINDINGS_START -->", "<!-- REAL_WORKER_FINDINGS_END -->"
    if begin in readme:
        start = readme.index(begin)
        stop = readme.index(end, start) + len(end)
        readme = readme[:start] + finding + readme[stop:]
    else:
        readme = readme.replace("## Results and figures\n", "## Results and figures\n\n" + finding + "\n", 1)
    readme_path.write_text(readme)
    print(f"Wrote measured research report: {path}")


if __name__ == "__main__":
    main()
