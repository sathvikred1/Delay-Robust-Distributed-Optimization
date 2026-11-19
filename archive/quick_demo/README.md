# Quick demonstration figures

The main presentation uses the full CoverType figures in `plots/real_workers/`.
The earlier Breast Cancer quick-run figures are kept here as one lossless archive
to avoid a second set of presentation images in the main plots folder.

The quick CSVs, models, event logs, and configuration remain in
`results/real_workers/quick/`. Each archived filename and checksum is listed in
`artifact_manifest.json`.

To regenerate the quick figures when needed:

```bash
python generate_real_worker_plots.py --results results/real_workers/quick --output plots/real_workers/quick
```

`figures.tar.gz` also preserves the original quick figures byte for byte.
