# Final presentation outline

Use the PNGs in `plots/real_workers/` and the corresponding vector PDFs for slides.
This is a talk outline; the report and CSVs contain the numerical evidence.

1. **Research question.** What changes when slow workers return outdated gradients?
   State single-machine scope and connect to the completed Week 9 simulation.
2. **Architecture and correctness.** Parent parameter server, read-only shared data,
   distinct worker PIDs, immutable model snapshots, queues, and measured wall time.
3. **Methods and fairness.** Four policies; threshold variants; unequal effective
   batches. Compare time, gradients, examples, and updates together.
4. **Data and experiment grid.** Breast Cancer demonstration; complete CoverType
   binary main task; fixed stratified splits; training-only scaler; three seeds;
   natural, moderate, and severe worker conditions.
5. **Convergence.** Use `01_training_loss_vs_time.png` and
   `02_validation_accuracy_vs_time.png`. Describe observed values from the report,
   not an expected winner. Show uncertainty and short-run limitations.
6. **Scaling, staleness, and filtering.** Use `03_worker_scaling.png`,
   `04_gradient_staleness.png`, and `05_threshold_comparison.png`.
   Explain discarded work and why mean age can hide extreme tail age.
7. **Work and timing.** Use `09_loss_vs_examples.png` and `10_worker_timeline.png`.
   Distinguish kernel computation, deliberate sleep, queue latency, evaluation,
   startup, and end-to-end costs.
8. **Evidence and conclusions.** Tests, full-grid completion, independent event
   audit, fixed seeds and source hashes. State the actual findings and limitations.
   Propose longer runs, matched batches, spatial holdouts, and a networked cluster.

When showing the historical simulation comparison, explicitly label synthetic data
and simulated units separately from real data and measured seconds.
