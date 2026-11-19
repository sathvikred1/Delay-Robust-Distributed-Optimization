# Week 9 simulation archive

This is the earlier simulation stage of our project. We kept it so the development
from simulated delays to real worker processes can be reviewed.

- [Week 9 report](week9_summary.md): the original full simulation report, unchanged.
- [Original README](original_readme.md): the earlier project description, unchanged.
- `simulation_artifacts.tar.gz`: the original full and quick CSV results, logs,
  verification files, figures, and the old development requirements file.
- `artifact_manifest.json`: original filenames, byte sizes, and SHA256 checksums.

The archive preserves each file's exact contents. Filenames mentioned in the old
report refer to the original folder structure stored inside this archive.
The simulation implementation remains in `src/`, `experiments/`, and `main.py`.
Its package lock is `requirements-lock.txt` at the repository root.

To inspect the old files, extract them into a separate empty directory:

```bash
mkdir -p /tmp/week9-simulation
tar -xzf archive/week9/simulation_artifacts.tar.gz -C /tmp/week9-simulation
```

To run a new simulation without mixing its output with the real-worker results:

```bash
python main.py --quick --output-dir results/simulation_runs/quick
python main.py --full --output-dir results/simulation_runs/full
```

The current simulation-comparison plot reads the historical CSV directly from the
archive. Extracting it is not required to generate the current figures.
