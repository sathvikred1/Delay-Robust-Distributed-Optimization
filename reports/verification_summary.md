# Checks and reproducibility

We checked the implementation at three levels: unit tests, the recorded worker
events, and the saved final models. The original simulation passed its 12 tests
before the extension. The current suite has **39 passing tests**; the terminal
output is in [test_results.txt](test_results.txt).

| Experiment | Runs | Returned gradients checked | Final models rescored |
| --- | --- | --- | --- |
| Full CoverType | 216 | 3,950,837 | 216 |
| Quick Breast Cancer | 36 | 230,461 | 36 |

The event checks cover task order, model snapshots, staleness, acceptance thresholds,
late arrivals, work counts, clean process exits, and the aggregate tables.
The compressed logs reconstruct the original event CSVs exactly.
The model check rebuilt each dataset split and recomputed all training,
validation, and test scores. They matched the saved results within numerical tolerance.

The dashboard passed its interaction checks and its local health endpoint returned
HTTP 200. Its tables and observations read the recorded CSVs.

## What we kept when organizing the repository

The 72 earlier simulation output/configuration files are preserved in
`archive/week9/simulation_artifacts.tar.gz`, with filenames and checksums in its
manifest. The Week 9 report and original README remain readable beside the archive.
The 23 quick-figure/provenance files are together in `archive/quick_demo/`.
Every archived file was checked against its original SHA256 before consolidation.

The main plots folder now contains the 11 full-experiment figures in PNG and PDF.
The quick figures can be restored from their archive or regenerated from the quick
results. The original simulation code and tests remain available. The comparison
plot reads the original simulation CSV directly from the archive.

No measured real-worker CSVs, models, event-log parts, source snapshots, or hardware
records were removed. We changed the presentation and organization, not the
experimental measurements.

## Scope of the evidence

These are short experiments on one machine. Arrival order depends on OS scheduling,
effective batches differ between methods, and three seeds support descriptive
mean/SD comparisons rather than statistical-significance claims.

To run the checks when needed:

```bash
python -m pytest -q
python scripts/real_worker_artifacts.py --results results/real_workers --verify
python scripts/verify_real_worker_models.py --results results/real_workers
```
