"""Publication-style figures from measured CSVs; no fitted performance claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from .artifacts import iter_event_chunks


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def result_directory(root=None):
    if root:
        return Path(root)
    base = Path(__file__).resolve().parents[1] / "results/real_workers"
    return base if (base / "raw_results.csv").exists() else base / "quick"


def curve_estimates(frame, x, y, points=100):
    """Interpolate each seed only over shared support; SD is across seeds."""
    traces = [g.sort_values("elapsed_seconds").drop_duplicates(x, keep="last").sort_values(x)
              for _, g in frame.groupby("run_id")]
    if not traces:
        return None
    low, high = max(t[x].min() for t in traces), min(t[x].max() for t in traces)
    if high <= low:
        return None
    grid = np.linspace(low, high, points)
    values = np.array([np.interp(grid, t[x], t[y]) for t in traces])
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else None
    return grid, values.mean(axis=0), sd


def draw_curves(ax, frame, x="elapsed_seconds", y="training_loss"):
    labels = sorted(frame.method_label.unique())
    for color, label in zip(COLORS, labels):
        data = frame[frame.method_label == label]
        estimate = curve_estimates(data, x, y)
        if estimate is None:
            continue
        grid, mean, sd = estimate
        ax.plot(grid, mean, label=label, color=color, lw=2)
        if sd is not None:
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.13)
    ax.grid(alpha=0.2)


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(results=None, output="plots/real_workers"):
    directory, output = result_directory(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(directory / "raw_results.csv")
    history = pd.read_csv(directory / "training_history.csv")
    manifest = json.loads((directory / "experiment_config.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Experiment is incomplete; finish or inspect it before generating final figures")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
                         "legend.fontsize": 9, "figure.constrained_layout.use": True})
    worker_count = int(raw.workers.max())
    focus = history[(history.workers == worker_count) & (history.scenario == "severe")]
    subtitle = f"{manifest['dataset']['name']} · {worker_count} workers · severe delay"
    for stem, x, y, xlabel, ylabel, title in [
        ("01_training_loss_vs_time", "elapsed_seconds", "training_loss", "Real training time (seconds)", "Training logistic loss", "Training loss vs real time"),
        ("02_validation_accuracy_vs_time", "elapsed_seconds", "validation_accuracy", "Real training time (seconds)", "Validation accuracy", "Validation accuracy vs real time"),
        ("08_loss_vs_gradients", "gradients_computed", "training_loss", "Gradient results received by server", "Training logistic loss", "Loss vs computed gradient work"),
        ("09_loss_vs_examples", "training_examples_processed", "training_loss", "Training examples processed (with repetition)", "Training logistic loss", "Loss vs examples processed"),
        ("11_loss_vs_updates", "optimizer_updates", "training_loss", "Applied optimizer updates", "Training logistic loss", "Loss vs optimizer updates"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_curves(ax, focus, x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{title}\n{subtitle}")
        ax.legend(loc="best")
        fig.supxlabel("Mean across seeds; shaded area = sample SD (where n > 1). Interpolated on shared support.", fontsize=9)
        save_figure(fig, output, stem)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, scenario in zip(axes, ("natural", "moderate", "severe")):
        for color, (label, frame) in zip(COLORS, raw[raw.scenario == scenario].groupby("method_label", sort=True)):
            grouped = frame.groupby("workers").worker_throughput.agg(["mean", "std"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0),
                        marker="o", capsize=3, label=label, color=color)
        ax.set(title=scenario.capitalize(), xlabel="OS worker processes", xticks=sorted(raw.workers.unique()))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("In-budget gradient results / second")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Worker count vs throughput · {manifest['dataset']['name']}")
    fig.supxlabel("Mean across seeds; error bars = sample SD where n > 1.", fontsize=9)
    save_figure(fig, output, "03_worker_scaling")

    subset = raw[(raw.workers == worker_count) & (raw.scenario == "severe")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, field, title in zip(axes, ("mean_staleness", "max_staleness"),
                                ("Mean arrival staleness", "Maximum arrival staleness")):
        grouped = subset.groupby("method_label")[field].agg(["mean", "std"])
        ax.bar(range(len(grouped)), grouped["mean"], yerr=grouped["std"].fillna(0), capsize=3, color=COLORS)
        ax.set_xticks(range(len(grouped)), grouped.index, rotation=40, ha="right")
        ax.set(title=title, ylabel="Applied updates since snapshot")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(subtitle + " · in-budget arrivals only")
    fig.supxlabel("Bars show seed means; error bars = sample SD where n > 1. Maximum = mean of run maxima.", fontsize=9)
    save_figure(fig, output, "04_gradient_staleness")

    filtered = subset[subset.method == "filtered"].copy()
    filtered["stale_percent"] = 100 * filtered.gradients_stale / filtered.gradients_computed_in_budget.clip(lower=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, field, ylabel in zip(axes, ("training_loss", "stale_percent"), ("Final training loss", "Stale gradients discarded (%)")):
        grouped = filtered.groupby("threshold")[field].agg(["mean", "std"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=4)
        ax.set(xlabel="Inclusive staleness threshold (updates)", ylabel=ylabel, xticks=[2, 5, 10])
        ax.grid(alpha=0.2)
    fig.suptitle("Staleness threshold comparison\n" + subtitle)
    fig.supxlabel("Mean across seeds; error bars = sample SD where n > 1.", fontsize=9)
    save_figure(fig, output, "05_threshold_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scenarios = ["natural", "moderate", "severe"]
    for color, (label, frame) in zip(COLORS, raw[raw.workers == worker_count].groupby("method_label", sort=True)):
        for ax, field in zip(axes, ("training_loss", "worker_throughput")):
            grouped = frame.groupby("scenario")[field].agg(["mean", "std"]).reindex(scenarios)
            ax.errorbar(range(3), grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=3, color=color, label=label)
    for ax, ylabel in zip(axes, ("Final training loss", "Gradient results / second")):
        ax.set_xticks(range(3), ["Natural", "+5 ms slow", "+25 ms slow"])
        ax.set(xlabel="Actual injected slow-worker sleep", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Effect of slow workers · {worker_count} workers")
    fig.supxlabel("Mean across seeds; error bars = sample SD where n > 1.", fontsize=9)
    save_figure(fig, output, "06_slow_workers")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    historical = Path(__file__).resolve().parents[1] / "results/raw_results.csv"
    if historical.exists():
        sim = pd.read_csv(historical)
        sim = sim[(sim.workers == 4) & (sim.scenario == "high")].copy()
        sim["elapsed_seconds"] = sim.simulated_time
        draw_curves(axes[0], sim, "simulated_time", "training_loss")
        axes[0].set(title="Week 9 simulation · synthetic data · 4 workers", xlabel="Simulated time units", ylabel="Training logistic loss")
    else:
        axes[0].text(0.5, 0.5, "Historical simulation CSV not available", ha="center")
    real = history[(history.workers == 4) & (history.scenario == "severe")]
    draw_curves(axes[1], real)
    axes[1].set(title="Real processes · real data · 4 workers", xlabel="Measured seconds", ylabel="Training logistic loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("Simulation and real-worker observations (different datasets and time units)")
    fig.supxlabel("Qualitative comparison only; these panels cannot quantify a real-world speedup.", fontsize=10)
    save_figure(fig, output, "07_simulation_vs_real")

    timeline_id = raw[(raw.workers == worker_count) & (raw.scenario == "severe") & (raw.method == "async")].sort_values("seed").iloc[0].run_id
    columns = ["run_id", "worker_id", "start_seconds", "finish_seconds", "arrival_seconds", "in_budget"]
    # Only load the selected timeline; large event logs do not need to fit in memory.
    chunks = [chunk[chunk.run_id == timeline_id] for chunk in iter_event_chunks(directory, usecols=columns)]
    events = pd.concat(chunks, ignore_index=True)
    horizon = min(0.2, float(raw.time_budget.iloc[0]))
    events = events[events.start_seconds <= horizon]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    zoom_start = horizon / 2
    for ax, (left, end) in zip(axes, ((0, horizon), (zoom_start, min(horizon, zoom_start + 0.004)))):
        for event in events[(events.start_seconds <= end) & (events.arrival_seconds >= left)].itertuples():
            ax.broken_barh([(event.start_seconds, event.finish_seconds - event.start_seconds)],
                          (event.worker_id - 0.25, 0.5), facecolors=COLORS[0], alpha=0.7)
            ax.plot(event.arrival_seconds, event.worker_id, "|", color=COLORS[1], markersize=9)
        ax.set(xlabel="Measured time since training start (seconds)", ylabel="OS worker ID",
               title=f"{left * 1000:g}–{end * 1000:g} ms (all tasks in view; long bars clipped to view)",
               yticks=range(worker_count), xlim=(left, end))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Real execution timeline · vanilla async · severe delay · seed {raw[raw.run_id == timeline_id].iloc[0].seed}")
    fig.supxlabel("Blue: worker active (computation + injected sleep). Orange ticks: server receives gradient.", fontsize=10)
    save_figure(fig, output, "10_worker_timeline")
    (output / "figure_provenance.json").write_text(json.dumps(dict(
        results=str(directory.relative_to(Path(__file__).resolve().parents[1])) if directory.is_relative_to(Path(__file__).resolve().parents[1]) else str(directory),
        dataset_sha256=manifest["dataset"]["sha256"], timeline_run_id=timeline_id,
        curves="Linear interpolation within shared seed support; bands are sample SD, not confidence intervals",
        figures=11), indent=2) + "\n")
    print(f"Saved 11 PNG and 11 PDF figures to {output}")
