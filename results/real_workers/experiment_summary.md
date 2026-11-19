# Measured real-worker experiment summary

Mode: **full**. Dataset: **covtype**. Completed 216 / 216 prespecified runs.

Single-machine multiprocessing; no networked cluster was tested. Data fingerprint: `8650e26a1d6811c2188929ff9cbfef3234c7600c4223966c48c0e873c234b929`.

Training budget: 3.0 seconds/run; startup and shutdown are measured separately. All models start at zero. Initial learning rate 0.05, sample batch 1024, L2=0.001; no outcome-based tuning.

## Observations from the largest worker count under severe delay

- Async-MB (B=4, tau=5): mean final training loss 0.5159, validation accuracy 0.757, 11389.9 received gradients/s.
- Synchronous: mean final training loss 0.5718, validation accuracy 0.710, 220.5 received gradients/s.
- Filtered (tau=10): mean final training loss 0.5160, validation accuracy 0.757, 10668.5 received gradients/s.
- Filtered (tau=5): mean final training loss 0.5159, validation accuracy 0.757, 10797.8 received gradients/s.
- Filtered (tau=2): mean final training loss 0.5160, validation accuracy 0.757, 10969.7 received gradients/s.
- Vanilla async: mean final training loss 0.5160, validation accuracy 0.756, 10474.9 received gradients/s.

These are descriptive measurements, not significance or universal speedup claims. A single quick seed has undefined sample SD. Full mode uses three seeds; OS scheduling and timing are not bitwise reproducible.

## Fairness and timing

Synchronous updates use workers × 1024 examples; async and filtered updates use 1024; mini-batch updates combine 4 × 1024. Sampling is with replacement from the same training set using independent reproducible per-worker streams. The arrival order differs. Compare loss against seconds, updates, gradients, and examples rather than update counts alone.

Staleness is the number of applied server updates since a task snapshot. Only in-budget arrivals enter mean, maximum, and p95 staleness and throughput. Gradients returned after stopping are drained, recorded as after_budget, and discarded. Accepted partial groups remain pending and are never applied for free. History counts mean results received by the server at that snapshot. Raw final counts also include draining.

Worker computation measures the gradient kernel including batch materialization, excluding injected sleep. Injected delays are actual worker sleeps: 5 ms moderate, 25 ms severe for the last max(1, floor(workers/4)) workers; natural adds no sleep. Communication/coordination is reported as dispatch cost, dispatch-to-start latency, finish-to-receipt latency, and parent coordination time. These overlap with computation; they are not additive CPU accounting or pure network cost. Queue wait also includes waiting for computation; it is not labeled communication overhead. Parent evaluation runs inside the training clock while workers continue. Final evaluation and draining are outside it.

## Limitations

Short local CPU experiments emphasize IPC and scheduling; additional workers can reduce throughput. Fixed learning rates and unequal effective batches limit optimizer comparisons. CoverType uses a declared binary one-vs-rest target; it is not seven-class classification. Random row splits can be optimistic for spatially correlated forest data. Held-out test scores are evaluated after training, never for preprocessing, stopping, or tuning. Repeated inspection of them must not guide subsequent tuning. Historical simulation uses synthetic data and simulated time, so the side-by-side figure is qualitative and cannot establish a real-world speedup.

## All configurations (means; sample SD is in aggregate_results.csv)

| Method | Workers | Delay | Train loss | Validation accuracy | Gradients/s | Mean / max staleness |
| --- | --- | --- | --- | --- | --- | --- |
| Vanilla async | 8 | moderate | 0.5160 | 0.756 | 10327.5 | 7.00 / 95.3 |
| Filtered (tau=5) | 2 | moderate | 0.5160 | 0.756 | 4187.7 | 0.96 / 30.0 |
| Filtered (tau=10) | 8 | natural | 0.5160 | 0.756 | 10575.8 | 6.64 / 23.7 |
| Filtered (tau=2) | 1 | natural | 0.5159 | 0.756 | 4112.3 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 8 | severe | 0.5159 | 0.757 | 11389.9 | 1.74 / 97.3 |
| Filtered (tau=10) | 4 | severe | 0.5159 | 0.757 | 9623.0 | 2.99 / 328.0 |
| Filtered (tau=10) | 2 | natural | 0.5159 | 0.756 | 7237.7 | 1.00 / 2.0 |
| Filtered (tau=5) | 8 | moderate | 0.5160 | 0.757 | 10991.8 | 5.76 / 81.7 |
| Filtered (tau=5) | 1 | natural | 0.5160 | 0.757 | 4075.5 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 1 | severe | 0.6273 | 0.697 | 31.0 | 0.00 / 0.0 |
| Vanilla async | 4 | moderate | 0.5159 | 0.757 | 9606.6 | 3.00 / 78.3 |
| Async-MB (B=4, tau=5) | 2 | moderate | 0.5159 | 0.756 | 4153.3 | 0.24 / 8.0 |
| Filtered (tau=2) | 2 | severe | 0.5160 | 0.756 | 4103.8 | 0.99 / 134.7 |
| Filtered (tau=10) | 1 | natural | 0.5159 | 0.757 | 3986.5 | 0.00 / 0.0 |
| Synchronous | 4 | natural | 0.5159 | 0.757 | 9870.8 | 0.00 / 0.0 |
| Vanilla async | 2 | moderate | 0.5160 | 0.756 | 3977.4 | 1.00 / 28.3 |
| Filtered (tau=2) | 8 | natural | 0.5160 | 0.757 | 10704.9 | 2.84 / 11.0 |
| Filtered (tau=5) | 2 | natural | 0.5160 | 0.757 | 6980.8 | 1.00 / 2.3 |
| Filtered (tau=2) | 4 | natural | 0.5159 | 0.757 | 10997.7 | 2.22 / 6.3 |
| Filtered (tau=5) | 2 | severe | 0.5159 | 0.757 | 4016.7 | 0.99 / 131.7 |
| Filtered (tau=10) | 8 | moderate | 0.5160 | 0.756 | 10222.4 | 6.78 / 88.7 |
| Synchronous | 8 | severe | 0.5718 | 0.710 | 220.5 | 0.00 / 0.0 |
| Vanilla async | 1 | severe | 0.5656 | 0.712 | 31.8 | 0.00 / 0.0 |
| Filtered (tau=5) | 1 | moderate | 0.5296 | 0.740 | 133.8 | 0.00 / 0.0 |
| Filtered (tau=5) | 1 | severe | 0.5655 | 0.712 | 31.9 | 0.00 / 0.0 |
| Filtered (tau=2) | 8 | moderate | 0.5159 | 0.757 | 10211.6 | 3.52 / 45.0 |
| Async-MB (B=4, tau=5) | 8 | moderate | 0.5159 | 0.757 | 10526.8 | 1.70 / 23.0 |
| Filtered (tau=2) | 1 | moderate | 0.5288 | 0.741 | 139.8 | 0.00 / 0.0 |
| Filtered (tau=2) | 2 | natural | 0.5160 | 0.756 | 7097.6 | 1.00 / 2.0 |
| Filtered (tau=10) | 2 | moderate | 0.5160 | 0.757 | 3900.2 | 0.96 / 29.0 |
| Synchronous | 4 | moderate | 0.5303 | 0.739 | 514.7 | 0.00 / 0.0 |
| Async-MB (B=4, tau=5) | 2 | natural | 0.5159 | 0.757 | 7047.6 | 0.25 / 1.0 |
| Filtered (tau=5) | 4 | severe | 0.5160 | 0.757 | 9333.8 | 2.98 / 324.3 |
| Async-MB (B=4, tau=5) | 1 | moderate | 0.5650 | 0.712 | 131.3 | 0.00 / 0.0 |
| Vanilla async | 1 | moderate | 0.5310 | 0.738 | 123.5 | 0.00 / 0.0 |
| Vanilla async | 4 | natural | 0.5160 | 0.757 | 10385.8 | 3.00 / 7.7 |
| Filtered (tau=2) | 4 | severe | 0.5159 | 0.756 | 9740.9 | 2.87 / 316.7 |
| Filtered (tau=10) | 1 | moderate | 0.5310 | 0.738 | 123.4 | 0.00 / 0.0 |
| Filtered (tau=2) | 2 | moderate | 0.5160 | 0.756 | 4161.9 | 0.96 / 29.7 |
| Synchronous | 1 | moderate | 0.5310 | 0.738 | 123.0 | 0.00 / 0.0 |
| Filtered (tau=10) | 4 | moderate | 0.5160 | 0.756 | 9445.9 | 2.95 / 78.3 |
| Filtered (tau=10) | 8 | severe | 0.5160 | 0.757 | 10668.5 | 6.92 / 368.7 |
| Async-MB (B=4, tau=5) | 8 | natural | 0.5159 | 0.757 | 10848.9 | 1.75 / 6.0 |
| Filtered (tau=5) | 4 | moderate | 0.5159 | 0.757 | 9328.8 | 2.95 / 76.0 |
| Synchronous | 2 | moderate | 0.5318 | 0.737 | 236.7 | 0.00 / 0.0 |
| Filtered (tau=10) | 1 | severe | 0.5669 | 0.711 | 30.7 | 0.00 / 0.0 |
| Filtered (tau=5) | 8 | natural | 0.5160 | 0.756 | 10430.7 | 4.91 / 19.0 |
| Async-MB (B=4, tau=5) | 1 | natural | 0.5159 | 0.756 | 4091.8 | 0.00 / 0.0 |
| Synchronous | 1 | severe | 0.5666 | 0.711 | 30.9 | 0.00 / 0.0 |
| Vanilla async | 2 | natural | 0.5159 | 0.757 | 7244.3 | 1.00 / 2.0 |
| Async-MB (B=4, tau=5) | 4 | moderate | 0.5159 | 0.757 | 10000.7 | 0.74 / 19.7 |
| Synchronous | 8 | natural | 0.5159 | 0.757 | 9995.0 | 0.00 / 0.0 |
| Synchronous | 1 | natural | 0.5159 | 0.756 | 3873.6 | 0.00 / 0.0 |
| Filtered (tau=5) | 4 | natural | 0.5160 | 0.757 | 10699.7 | 2.99 / 7.0 |
| Filtered (tau=5) | 8 | severe | 0.5159 | 0.757 | 10797.8 | 5.86 / 315.0 |
| Async-MB (B=4, tau=5) | 2 | severe | 0.5159 | 0.757 | 4101.0 | 0.25 / 33.7 |
| Async-MB (B=4, tau=5) | 4 | severe | 0.5159 | 0.756 | 10032.7 | 0.75 / 84.7 |
| Filtered (tau=2) | 8 | severe | 0.5160 | 0.757 | 10969.7 | 3.57 / 197.7 |
| Synchronous | 2 | natural | 0.5159 | 0.757 | 6587.9 | 0.00 / 0.0 |
| Filtered (tau=10) | 2 | severe | 0.5160 | 0.757 | 3809.3 | 0.99 / 124.7 |
| Vanilla async | 8 | severe | 0.5160 | 0.756 | 10474.9 | 6.98 / 391.3 |
| Vanilla async | 8 | natural | 0.5159 | 0.757 | 10322.3 | 7.00 / 24.3 |
| Filtered (tau=2) | 1 | severe | 0.5666 | 0.711 | 30.9 | 0.00 / 0.0 |
| Synchronous | 4 | severe | 0.5685 | 0.712 | 119.3 | 0.00 / 0.0 |
| Vanilla async | 2 | severe | 0.5160 | 0.757 | 4001.9 | 1.00 / 131.3 |
| Async-MB (B=4, tau=5) | 4 | natural | 0.5159 | 0.757 | 11346.4 | 0.75 / 2.0 |
| Vanilla async | 4 | severe | 0.5159 | 0.756 | 9688.8 | 2.99 / 341.0 |
| Vanilla async | 1 | natural | 0.5160 | 0.756 | 4046.3 | 0.00 / 0.0 |
| Synchronous | 8 | moderate | 0.5325 | 0.736 | 910.5 | 0.00 / 0.0 |
| Filtered (tau=2) | 4 | moderate | 0.5159 | 0.757 | 8969.3 | 2.82 / 86.7 |
| Filtered (tau=10) | 4 | natural | 0.5160 | 0.757 | 10530.4 | 3.00 / 7.3 |
| Synchronous | 2 | severe | 0.5666 | 0.712 | 62.3 | 0.00 / 0.0 |
