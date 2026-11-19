"""Lossless, bounded-size event archives for Git; verification independent of run()."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_event_chunks(directory, **kwargs):
    directory = Path(directory)
    if (directory / "gradient_events.csv").exists():
        yield from pd.read_csv(directory / "gradient_events.csv", chunksize=50000, **kwargs)
    else:
        index = json.loads((directory / "gradient_event_parts/index.json").read_text())
        for part in index["parts"]:
            yield from pd.read_csv(directory / "gradient_event_parts" / part["file"], chunksize=50000, **kwargs)


def snapshot_sources(directory):
    directory = Path(directory)
    config = directory / "experiment_config.json"
    if not config.exists():
        return
    manifest = json.loads(config.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in manifest["source_sha256"].items():
        target = directory / "source_snapshot" / name
        source = target if target.exists() else root / name
        if sha256(source) != expected:
            raise ValueError(f"Source bytes no longer match run manifest: {name}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())


def archive_events(directory, rows_per_part=50000):
    directory = Path(directory)
    snapshot_sources(directory)
    source = directory / "gradient_events.csv"
    archive = directory / "gradient_event_parts"
    if (archive / "index.json").exists():
        index = json.loads((archive / "index.json").read_text())
        if sha256(source) != index["csv_sha256"]:
            raise ValueError("Existing archive differs; use a new result directory")
        return index
    archive.mkdir(exist_ok=True)
    parts, rows = [], 0
    # CSV values in this schema contain no embedded newlines. Byte-preserving
    # partitioning means restore produces exactly the original CSV, including floats.
    with source.open("rb") as stream:
        header = stream.readline()
        while True:
            lines = []
            for _ in range(rows_per_part):
                line = stream.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            path = archive / f"part-{len(parts):04d}.csv.gz"
            with path.open("wb") as dest:
                with gzip.GzipFile(filename="", mode="wb", fileobj=dest, mtime=0) as zipped:
                    zipped.write(header)
                    zipped.writelines(lines)
            parts.append(dict(file=path.name, rows=len(lines), sha256=sha256(path)))
            rows += len(lines)
    index = dict(csv_sha256=sha256(source), rows=rows, parts=parts,
                 format="Each gzip member is a CSV with header; restore omits repeated headers")
    (archive / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def restore_events(directory):
    directory = Path(directory)
    archive = directory / "gradient_event_parts"
    index = json.loads((archive / "index.json").read_text())
    target = directory / "gradient_events.csv"
    if target.exists():
        if sha256(target) != index["csv_sha256"]:
            raise ValueError("Existing CSV differs; refusing to overwrite")
        return target
    temporary = directory / "gradient_events.csv.restoring"
    with temporary.open("wb") as output:
        for number, part in enumerate(index["parts"]):
            path = archive / part["file"]
            if sha256(path) != part["sha256"]:
                raise ValueError(f"Corrupt event archive: {path}")
            with gzip.open(path, "rb") as source:
                header = source.readline()
                if number == 0:
                    output.write(header)
                for line in source:
                    output.write(line)
    if sha256(temporary) != index["csv_sha256"]:
        raise ValueError("Restored CSV hash mismatch")
    temporary.rename(target)
    return target


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "experiment_config.json").read_text())
    raw = pd.read_csv(directory / "raw_results.csv")
    history = pd.read_csv(directory / "training_history.csv")
    timing = pd.read_csv(directory / "worker_timing.csv")
    totals = {}
    errors = []
    seen = set()
    snapshot_hashes = {}
    for frame in iter_event_chunks(directory):
        for run_id, e in frame.groupby("run_id"):
            state = totals.setdefault(run_id, dict(computed=0, accepted=0, used_updates=0,
                in_budget=0, stale=0, after=0, compute_seconds=0.0, staleness_sum=0, max_staleness=0))
            state["computed"] += len(e)
            state["accepted"] += int(e.accepted.sum())
            state["used_updates"] += int(e.update_applied.sum())
            state["in_budget"] += int(e.in_budget.sum())
            state["stale"] += int((e.reason == "stale").sum())
            state["after"] += int((e.reason == "after_budget").sum())
            state["compute_seconds"] += float(e.compute_seconds.sum())
            state["staleness_sum"] += int(e.loc[e.in_budget, "staleness"].sum())
            state["max_staleness"] = max(state["max_staleness"], int(e.loc[e.in_budget, "staleness"].max()) if e.in_budget.any() else 0)
        if not (frame.staleness == frame.arrival_version - frame.snapshot_version).all():
            errors.append("Incorrect event staleness")
        if not (frame.version_after == frame.arrival_version + frame.update_applied.astype(int)).all():
            errors.append("Incorrect model-version transition")
        if (frame.accepted & ~frame.in_budget).any():
            errors.append("After-budget gradient applied")
        if not (frame.start_seconds <= frame.finish_seconds).all() or not (frame.finish_seconds <= frame.arrival_seconds).all():
            errors.append("Nonmonotonic process timestamps")
        # Per-snapshot fingerprints must agree even when different workers return it.
        for row in frame[["run_id", "snapshot_version", "snapshot_sha256"]].drop_duplicates().itertuples(index=False):
            key = (row.run_id, row.snapshot_version)
            if key in snapshot_hashes and snapshot_hashes[key] != row.snapshot_sha256:
                errors.append("Snapshot version has inconsistent parameter fingerprint")
            snapshot_hashes[key] = row.snapshot_sha256
        for row in frame[["run_id", "worker_id", "task_id"]].itertuples(index=False, name=None):
            if row in seen:
                errors.append("Duplicate worker task")
            seen.add(row)
    for row in raw.itertuples():
        e = totals[row.run_id]
        pairs = [(row.gradients_computed, e["computed"]), (row.gradients_accepted, e["accepted"]),
                 (row.optimizer_updates, e["used_updates"]), (row.gradients_computed_in_budget, e["in_budget"]),
                 (row.gradients_stale, e["stale"]), (row.gradients_after_budget, e["after"]),
                 (row.gradients_started, e["computed"]),
                 (row.gradients_computed, row.gradients_accepted + row.gradients_discarded),
                 (row.gradients_accepted, row.gradients_used + row.gradients_pending),
                 (row.training_examples_processed, e["computed"] * row.sample_batch_size),
                 (row.optimizer_updates * row.effective_batch_size, row.gradients_used * row.sample_batch_size),
                 (row.max_staleness, e["max_staleness"])]
        if any(a != b for a, b in pairs):
            errors.append(f"Accounting mismatch: {row.run_id}")
        if not np.isclose(row.mean_staleness, e["staleness_sum"] / max(1, e["in_budget"])):
            errors.append(f"Mean staleness mismatch: {row.run_id}")
        final = history[history.run_id == row.run_id].iloc[-1]
        if not np.isclose(row.training_loss, final.training_loss) or row.optimizer_updates != final.optimizer_updates:
            errors.append(f"History endpoint mismatch: {row.run_id}")
        workers = timing[timing.run_id == row.run_id]
        if len(workers) != row.workers or workers.gradients_computed.sum() != e["computed"] or not workers.exitcode.eq(0).all():
            errors.append(f"Worker timing/shutdown mismatch: {row.run_id}")
        if not np.isclose(workers.worker_compute_seconds.sum(), e["compute_seconds"]):
            errors.append(f"Worker computation sum mismatch: {row.run_id}")
    root = Path(__file__).resolve().parents[1]
    def recorded_source(path):
        snapshot = directory / "source_snapshot" / path
        return snapshot if snapshot.exists() else root / path

    mismatches = [path for path, digest in manifest["source_sha256"].items()
                  if sha256(recorded_source(path)) != digest]
    current_differences = [path for path, digest in manifest["source_sha256"].items()
                           if sha256(root / path) != digest]
    # A byte-exact source snapshot keeps older measurements auditable when tooling evolves.
    if mismatches:
        errors.append(f"Experiment source changed since run: {mismatches}")
    if not manifest["complete"] or len(raw) != len(manifest["configurations"]):
        errors.append("Prespecified grid incomplete")
    if raw.run_id.duplicated().any():
        errors.append("Duplicate run IDs")
    from .experiments import aggregate
    expected = aggregate(raw)
    actual = pd.read_csv(directory / "aggregate_results.csv")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    except AssertionError:
        errors.append("Aggregate CSV differs from independent recalculation")
    index_path = directory / "gradient_event_parts/index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for part in index["parts"]:
            if sha256(index_path.parent / part["file"]) != part["sha256"]:
                errors.append(f"Archive checksum failed: {part['file']}")
        if (directory / "gradient_events.csv").exists() and sha256(directory / "gradient_events.csv") != index["csv_sha256"]:
            errors.append("Archive and original CSV differ")
    report = dict(passed=not errors, runs_checked=len(raw), events_checked=sum(t["computed"] for t in totals.values()),
                  snapshot_versions_checked=len(snapshot_hashes), source_hashes_match=not mismatches,
                  current_source_differences=current_differences,
                  errors=errors, checks=["grid completeness", "event uniqueness", "snapshot fingerprints",
                      "staleness arithmetic", "version transitions", "budget exclusion", "event timing",
                      "gradient/example/update accounting", "worker timing and clean exits",
                      "history endpoints", "independent aggregation", "source and archive SHA256"])
    (directory / "artifact_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise AssertionError(errors)
    return report
