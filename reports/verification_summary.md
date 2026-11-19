# Verification summary

The baseline simulation passed all 12 original tests before changes.
The final complete suite passed 38 tests from the permanent worktree. `python -m pip check` also passed. Actual terminal output is retained in
[test_results.txt](test_results.txt).

| Suite | Completed runs | Returned gradients audited | Models independently rescored | Checks |
| --- | --- | --- | --- | --- |
| Full CoverType | 216 | 3,950,837 | 216 | passed |
| Quick Breast Cancer | 36 | 230,461 | 36 | passed |

The event audit verifies task sequences, snapshot-version fingerprints, staleness,
model versions, acceptance thresholds, budget exclusion, example/gradient/update
counts, clean worker exits, independent aggregation, source snapshots, and lossless
reconstruction of compressed event logs. The model audit rebuilt each real dataset
split, reproduced its fingerprint, and recomputed training, validation, and test
scores for every saved model.

The Streamlit server was launched on localhost and its /_stcore/health endpoint
returned HTTP 200 with body `ok`. AppTest also exercised suite selection, the
examples-processed axis, worker filtering, and the empty-selection state.

Both suites have 11 scientific figures, each saved as PNG and PDF: 44 image files.
The representative loss, scaling, simulation comparison, and timeline images were
visually inspected for labeling and layout.

Only README.md, requirements.txt, and .gitignore changed among previously tracked
files. The original simulation source, experiments, tests, reports, CSVs, and plots
remain unchanged. The original README text is retained inside the updated README.

Limitations: execution is single-machine multiprocessing. Budgets are short and
arrival order is OS-dependent. Three seeds support descriptive mean/SD summaries,
not significance claims. Effective batches differ by algorithm. Event CSVs are
saved locally and versioned as exact compressed parts to respect GitHub file limits.

To reproduce the audits:

```bash
python -m pytest -q
python scripts/real_worker_artifacts.py --results results/real_workers --verify
python scripts/verify_real_worker_models.py --results results/real_workers
```
