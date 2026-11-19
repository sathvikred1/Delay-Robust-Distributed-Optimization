# Measured real-worker experiment summary

Mode: **quick**. Dataset: **breast_cancer**. Completed 36 / 36 prespecified runs.

Single-machine multiprocessing; no networked cluster was tested. Data fingerprint: `d89439d8a86d67b65141c075a016a0ea4a0f7700eba56c85d6b6c487068b626e`.

Training budget: 0.8 seconds/run; startup and shutdown are measured separately. All models start at zero. Initial learning rate 0.05, sample batch 128, L2=0.001; no outcome-based tuning.

## Observations from the largest worker count under severe delay

- Vanilla async: mean final training loss 0.0719, validation accuracy 0.988, 13790.5 received gradients/s.
- Synchronous: mean final training loss 0.2432, validation accuracy 0.976, 128.5 received gradients/s.
- Async-MB (B=4, tau=5): mean final training loss 0.0774, validation accuracy 1.000, 15902.6 received gradients/s.
- Filtered (tau=5): mean final training loss 0.0719, validation accuracy 0.988, 14200.4 received gradients/s.
- Filtered (tau=10): mean final training loss 0.0719, validation accuracy 0.988, 14134.2 received gradients/s.
- Filtered (tau=2): mean final training loss 0.0721, validation accuracy 0.988, 14414.9 received gradients/s.

These are descriptive measurements, not significance or universal speedup claims. A single quick seed has undefined sample SD. Full mode uses three seeds; OS scheduling and timing are not bitwise reproducible.

## Fairness and timing

Synchronous updates use workers × 128 examples; async and filtered updates use 128; mini-batch updates combine 4 × 128. Sampling is with replacement from the same training set using independent reproducible per-worker streams. The arrival order differs. Compare loss against seconds, updates, gradients, and examples rather than update counts alone.

Staleness is the number of applied server updates since a task snapshot. Only in-budget arrivals enter mean, maximum, and p95 staleness and throughput. Gradients returned after stopping are drained, recorded as after_budget, and discarded. Accepted partial groups remain pending and are never applied for free. History counts mean results received by the server at that snapshot. Raw final counts also include draining.

Worker computation measures the gradient kernel including batch materialization, excluding injected sleep. Injected delays are actual worker sleeps: 5 ms moderate, 25 ms severe for the last max(1, floor(workers/4)) workers; natural adds no sleep. Communication/coordination is reported as dispatch cost, dispatch-to-start latency, finish-to-receipt latency, and parent coordination time. These overlap with computation; they are not additive CPU accounting or pure network cost. Queue wait also includes waiting for computation; it is not labeled communication overhead. Parent evaluation runs inside the training clock while workers continue. Final evaluation and draining are outside it.

## Limitations

Short local CPU experiments emphasize IPC and scheduling; additional workers can reduce throughput. Fixed learning rates and unequal effective batches limit optimizer comparisons. CoverType uses a declared binary one-vs-rest target; it is not seven-class classification. Random row splits can be optimistic for spatially correlated forest data. Held-out test scores are evaluated after training, never for preprocessing, stopping, or tuning. Repeated inspection of them must not guide subsequent tuning. Historical simulation uses synthetic data and simulated time, so the side-by-side figure is qualitative and cannot establish a real-world speedup.

## All configurations (means; sample SD is in aggregate_results.csv)

| Method | Workers | Delay | Train loss | Validation accuracy | Gradients/s | Mean / max staleness |
| --- | --- | --- | --- | --- | --- | --- |
| Filtered (tau=10) | 1 | moderate | 0.1419 | 0.988 | 144.8 | 0.00 / 0.0 |
| Vanilla async | 4 | severe | 0.0719 | 0.988 | 13790.5 | 2.98 / 430.0 |
| Vanilla async | 1 | severe | 0.2395 | 0.988 | 32.5 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 4 | moderate | 0.0775 | 1.000 | 15781.1 | 0.74 / 27.0 |
| Synchronous | 4 | severe | 0.2432 | 0.976 | 128.5 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 4 | severe | 0.0774 | 1.000 | 15902.6 | 0.75 / 123.0 |
| Filtered (tau=5) | 1 | severe | 0.2395 | 0.988 | 32.4 | 0.00 / 0.0 |
| Filtered (tau=5) | 1 | natural | 0.0732 | 0.988 | 8872.0 | 0.00 / 0.0 |
| Synchronous | 1 | severe | 0.2395 | 0.988 | 32.4 | 0.00 / 0.0 |
| Filtered (tau=2) | 1 | natural | 0.0733 | 0.988 | 8642.1 | 0.00 / 0.0 |
| Vanilla async | 4 | moderate | 0.0718 | 0.988 | 14238.3 | 3.00 / 96.0 |
| Vanilla async | 1 | moderate | 0.1412 | 0.988 | 147.3 | 0.00 / 0.0 |
| Filtered (tau=10) | 1 | natural | 0.0733 | 0.988 | 8668.2 | 0.00 / 0.0 |
| Filtered (tau=2) | 1 | severe | 0.2395 | 0.988 | 32.5 | 0.00 / 0.0 |
| Filtered (tau=2) | 4 | natural | 0.0727 | 0.988 | 14404.8 | 2.14 / 8.0 |
| Filtered (tau=2) | 4 | moderate | 0.0721 | 0.988 | 14421.0 | 2.64 / 89.0 |
| Filtered (tau=5) | 4 | severe | 0.0719 | 0.988 | 14200.4 | 2.97 / 435.0 |
| Async-MB (B=4, tau=5) | 1 | severe | 0.4071 | 0.953 | 32.4 | 0.00 / 0.0 |
| Filtered (tau=10) | 4 | moderate | 0.0720 | 0.988 | 13312.9 | 2.96 / 96.0 |
| Synchronous | 4 | natural | 0.0770 | 1.000 | 16864.5 | 0.00 / 0.0 |
| Filtered (tau=5) | 4 | moderate | 0.0719 | 0.988 | 13974.8 | 2.96 / 95.0 |
| Filtered (tau=10) | 1 | severe | 0.2395 | 0.988 | 32.4 | 0.00 / 0.0 |
| Filtered (tau=10) | 4 | severe | 0.0719 | 0.988 | 14134.2 | 2.97 / 440.0 |
| Synchronous | 4 | moderate | 0.1439 | 0.988 | 558.4 | 0.00 / 0.0 |
| Vanilla async | 4 | natural | 0.0719 | 0.988 | 13989.6 | 3.00 / 9.0 |
| Filtered (tau=5) | 1 | moderate | 0.1422 | 0.988 | 143.6 | 0.00 / 0.0 |
| Synchronous | 1 | moderate | 0.1415 | 0.988 | 146.0 | 0.00 / 0.0 |
| Synchronous | 1 | natural | 0.0733 | 0.988 | 8681.3 | 0.00 / 0.0 |
| Filtered (tau=5) | 4 | natural | 0.0719 | 0.988 | 14059.7 | 2.94 / 9.0 |
| Async-MB (B=4, tau=5) | 1 | natural | 0.0818 | 1.000 | 9135.5 | 0.00 / 0.0 |
| Filtered (tau=2) | 1 | moderate | 0.1419 | 0.988 | 144.8 | 0.00 / 0.0 |
| Filtered (tau=2) | 4 | severe | 0.0721 | 0.988 | 14414.9 | 2.67 / 399.0 |
| Async-MB (B=4, tau=5) | 4 | natural | 0.0775 | 1.000 | 15691.2 | 0.75 / 3.0 |
| Vanilla async | 1 | natural | 0.0732 | 0.988 | 8936.4 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 1 | moderate | 0.2296 | 0.988 | 144.8 | 0.00 / 0.0 |
| Filtered (tau=10) | 4 | natural | 0.0719 | 0.988 | 14086.1 | 3.00 / 9.0 |
