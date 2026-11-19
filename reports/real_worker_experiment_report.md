# Real-worker experiment report

CSCI 8000 — Optimization in AI. This report extends the completed Week 9 simulation.
It describes the new measured implementation and does not change the historical
simulation report or its interpretation.

## Research objective

Study how delays and stale gradients change progress, computation use, and throughput
for synchronous SGD, vanilla asynchronous SGD, delay-filtered asynchronous SGD, and
simplified asynchronous mini-batch SGD. The research question is about the interaction
between optimization and execution, not simply which method completes more updates.

## System and correctness

One parent process is the parameter server. It alone owns and updates the model.
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

## Data and reproducible preparation

The quick suite uses the real Breast Cancer Wisconsin Diagnostic dataset and
completed 36 runs. It is a demonstration with one seed, not a statistical
benchmark. The main suite uses all 581,012 Forest CoverType
rows and 54 input features. Its declared binary target is
Lodgepole Pine (original class 2) versus all other forest cover types.

The main split has 406,708 training,
87,152 validation, and 87,152 test
rows. Splits are stratified and fixed using seeds 2027 and 2028. Preprocessing is
fitted on training rows only. The scaler is then applied to validation and test data,
and an intercept column is appended. Test data is evaluated after training and never
used for preprocessing, stopping, or parameter selection. All models initialize at zero.

Dataset fingerprint: `8650e26a1d6811c2188929ff9cbfef3234c7600c4223966c48c0e873c234b929`.
The manifest also records split-index fingerprints, scaler parameters, and class
balance. Random row splits may be optimistic for spatially correlated forest data.

## Experimental protocol

The completed full grid contains **216 runs**: 1, 2, 4, and 8 workers; natural,
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

## Hardware and verification

Recorded machine: Apple M2,
8 logical CPUs,
8 GiB RAM, macOS-26.2-arm64-arm-64bit-Mach-O.
Python: 3.14.3. Native BLAS threads are limited to one per
process. Complete package versions are in hardware_info.json and the new lock file.

The full suite took 1023.2 measured seconds, excluding data
loading and subsequent artifact processing. The independent audit checked
**3,950,837 returned gradients** and
**2,930,207 model-version fingerprints** across
all 216 runs. It checks cross-file accounting, time order,
clean exits, history endpoints, independently recalculated aggregates, and source
and archive checksums. Byte-exact source snapshots preserve the code used by older
measurements even when reporting/provenance tools evolve.

A separate model audit rebuilt the dataset split, reproduced its exact fingerprint,
and recomputed training, validation, and test scores for all
216 saved full-run models. These matched the
measured CSVs within numerical tolerance. The quick models passed the same check.

## Main measured comparison: 8 workers, severe delays

Values are mean ± sample SD across three seeds. Stale-discard percentages exclude
post-budget draining; maximum age is the mean of per-run maxima, not the pooled max.

| Method | Train loss | Val accuracy | Gradients/s | Updates | Stale discard % | Mean / max age |
| --- | --- | --- | --- | --- | --- | --- |
| Async-MB (B=4, tau=5) | 0.51588 ± 0.00000 | 0.757 ± 0.000 | 11389.9 ± 200.6 | 8494.7 ± 150.4 | 0.56 | 1.74 / 97.3 |
| Filtered (tau=10) | 0.51595 ± 0.00002 | 0.757 ± 0.000 | 10668.5 ± 233.6 | 31706.7 ± 688.0 | 0.94 | 6.92 / 368.7 |
| Filtered (tau=2) | 0.51596 ± 0.00001 | 0.757 ± 0.000 | 10969.7 ± 137.4 | 16793.0 ± 206.1 | 48.97 | 3.57 / 197.7 |
| Filtered (tau=5) | 0.51594 ± 0.00004 | 0.757 ± 0.000 | 10797.8 ± 49.6 | 27186.7 ± 114.9 | 16.07 | 5.86 / 315.0 |
| Synchronous | 0.57183 ± 0.00509 | 0.710 ± 0.002 | 220.5 ± 28.2 | 82.0 ± 10.5 | 0.00 | 0.00 / 0.0 |
| Vanilla async | 0.51596 ± 0.00002 | 0.756 ± 0.000 | 10474.9 ± 273.9 | 31425.0 ± 821.9 | 0.00 | 6.98 / 391.3 |

All 216/216 main runs ended below their zero-model training loss of
0.6931. This is a convergence sanity check, not proof of optimizer superiority.

Under severe delay, synchronous SGD received
220.5 gradients/s and completed
82.0 updates on average. Under natural execution
at the same worker count it received 9995.0
gradients/s and completed 3748.0 updates.
Vanilla async under severe delay received 10474.9
gradients/s, with mean final training loss 0.51596.
Faster workers can continue while slow tasks are outstanding; effective batches and
amounts of work still differ, so this is not a batch-matched optimizer comparison.

## Worker counts under natural execution

| Workers | Gradients/s | Train loss | Mean / max age |
| --- | --- | --- | --- |
| 1 | 4046.3 ± 25.5 | 0.5160 ± 0.0000 | 0.00 / 0.0 |
| 2 | 7244.3 ± 150.7 | 0.5159 ± 0.0000 | 1.00 / 2.0 |
| 4 | 10385.8 ± 230.7 | 0.5160 ± 0.0000 | 3.00 / 7.7 |
| 8 | 10322.3 ± 246.2 | 0.5159 ± 0.0000 | 7.00 / 24.3 |

These are local throughput measurements, not a linear-scaling claim. More processes
share the CPU and memory system and increase queue and server work. A small final-loss
difference after many gradients is not enough to establish a better optimizer.

At 8 workers, vanilla async's mean staleness changed from
7.00 under natural execution to
6.98 under severe delay. Its mean maximum staleness
changed from 24.3 to
391.3. Report both central and tail age: a modest mean
does not rule out very old gradients. Mini-batch age is counted in larger model
updates, so its update-based staleness is not directly batch-matched to vanilla async.

## Filtering thresholds

| Threshold | Train loss | Val accuracy | Stale discard % |
| --- | --- | --- | --- |
| 2 | 0.51596 ± 0.00001 | 0.7568 ± 0.0005 | 48.97 ± 0.03 |
| 5 | 0.51594 ± 0.00004 | 0.7567 ± 0.0003 | 16.07 ± 0.10 |
| 10 | 0.51595 ± 0.00002 | 0.7568 ± 0.0001 | 0.94 ± 0.03 |

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

## Limitations and next experiments

The budgets are short and the model is convex binary logistic regression. Natural
local overhead can dominate small gradient tasks. Some runs occurred while plots,
tests, or other desktop activity were running; the machine was not exclusively
reserved. Fixed randomized run order reduces systematic order bias but does not
remove background-load effects. Only one machine, three seeds, one main dataset,
fixed learning rates, unequal batches, and fixed slow-worker sleeps were tested.
Test scores are retained for transparency, but repeated inspection must not guide
future tuning on the same held-out data.

The results support a measured study of synchronization, stale work, and local
coordination, not universal superiority or multi-machine optimality. Next experiments
should use longer quiet-machine runs, more seeds, matched effective batches and
common tuning budgets, spatial holdouts, time-varying worker speeds, and physical
networked machines. The fixed mini-batch and fixed-threshold methods are educational
simplifications of the research papers, not exact reproductions of their guarantees.

## References and reproduction

- Attia, Gaash, Koren, ICML 2025: https://proceedings.mlr.press/v267/attia25a.html
- Maranjyan, Tyurin, Richtárik, ICML 2025: https://proceedings.mlr.press/v267/maranjyan25b.html
- Breast Cancer data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- CoverType data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html

See README.md for installation, exact commands, event-log restoration, and protected
output directories. Use reports/professor_demo.md for the 5–7 minute demonstration
and reports/presentation_outline.md to prepare classroom slides.
