"""Run from the repository root: streamlit run dashboard/app.py."""
import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from real_workers.plotting import curve_estimates

st.set_page_config(page_title="Delay-Robust Distributed Optimization", layout="wide")
st.title("Delay-Robust Distributed Optimization")
st.write("CSCI 8000 · How worker delays and gradient staleness affect SGD")
st.info("These are real Python worker processes on one machine. This experiment does not run across separate physical machines.")

base = ROOT / "results/real_workers"
available = [p for p in [base, base / "quick"] if (p / "raw_results.csv").exists()]
if not available:
    st.warning("No measured results found. Run: python run_real_workers.py --quick")
    st.stop()
choice = st.sidebar.selectbox("Experiment", available, format_func=lambda p: "Full · Forest CoverType" if p == base else "Quick · Breast Cancer")
manifest = json.loads((choice / "experiment_config.json").read_text())
hardware = json.loads((choice / "hardware_info.json").read_text())
if not manifest.get("complete"):
    st.warning(f"Experiment in progress or interrupted: {manifest['completed_runs']} runs saved. Conclusions are provisional.")


@st.cache_data
def load_csv(path, modified):
    return pd.read_csv(path)


raw = load_csv(str(choice / "raw_results.csv"), (choice / "raw_results.csv").stat().st_mtime_ns)
history = load_csv(str(choice / "training_history.csv"), (choice / "training_history.csv").stat().st_mtime_ns)
workers = st.sidebar.multiselect("Workers", sorted(raw.workers.unique()), default=sorted(raw.workers.unique()))
scenarios = st.sidebar.multiselect("Delay scenarios", ["natural", "moderate", "severe"], default=["severe"])
methods = st.sidebar.multiselect("Methods", sorted(raw.method_label.unique()), default=sorted(raw.method_label.unique()))
seeds = st.sidebar.multiselect("Seeds", sorted(raw.seed.unique()), default=sorted(raw.seed.unique()))
selected = raw[raw.workers.isin(workers) & raw.scenario.isin(scenarios) & raw.method_label.isin(methods) & raw.seed.isin(seeds)]
curves = history[history.run_id.isin(selected.run_id)]
if selected.empty:
    st.warning("Select at least one available combination in the sidebar.")
    st.stop()

st.subheader("Dataset and experiment")
data = manifest["dataset"]
a, b, c, d = st.columns(4)
a.metric("Dataset", data["name"])
b.metric("Training rows", f"{data['train_samples']:,}")
c.metric("Validation rows", f"{data['validation_samples']:,}")
d.metric("Test rows", f"{data['test_samples']:,}")
st.write(data["target"])
st.caption(f"{data['split_ratio']}; split seed {data['split_seed']}. Scaling fitted on training only. "
           f"Sampling with replacement; zero initial model. Completed runs: {manifest['completed_runs']}.")
with st.expander("Dataset provenance and preprocessing"):
    st.json(data)

st.subheader("Algorithm comparison")
st.dataframe(selected.groupby("method_label").agg(
    runs=("run_id", "count"), mean_train_loss=("training_loss", "mean"),
    mean_validation_accuracy=("validation_accuracy", "mean"),
    mean_gradients_per_second=("worker_throughput", "mean"),
    mean_optimizer_updates=("optimizer_updates", "mean")), width="stretch")
st.caption("Table means pool only the selected configurations. Narrow the filters for comparisons at fixed worker count and delay.")
st.write("Synchronous SGD averages one gradient from every worker. Vanilla async applies each arrival. "
         "Filtered async accepts staleness ≤ threshold. Simplified async mini-batch averages four accepted gradients per update.")
st.dataframe(selected[["method_label", "effective_batch_size"]].drop_duplicates(), hide_index=True)

st.subheader("Training loss and validation accuracy")
axis_name = st.selectbox("Compare progress against", ["Real training time", "Optimizer updates", "Gradients computed", "Training examples processed"])
x_field = {"Real training time": "elapsed_seconds", "Optimizer updates": "optimizer_updates",
           "Gradients computed": "gradients_computed", "Training examples processed": "training_examples_processed"}[axis_name]


def plot_curves(metric, title):
    rows = []
    for (method, count, scenario), frame in curves.groupby(["method_label", "workers", "scenario"]):
        estimate = curve_estimates(frame, x_field, metric)
        if estimate is None:
            continue
        grid, mean, sd = estimate
        for i in range(len(grid)):
            rows.append(dict(progress=grid[i], value=mean[i],
                             lower=mean[i] - (sd[i] if sd is not None else 0),
                             upper=mean[i] + (sd[i] if sd is not None else 0),
                             series=f"{method} · {count} workers · {scenario}"))
    if not rows:
        st.caption("Not enough progress for this axis in the selected runs.")
        return
    chart_data = pd.DataFrame(rows)
    base_chart = alt.Chart(chart_data).encode(x=alt.X("progress:Q", title=axis_name),
        color=alt.Color("series:N", title="Configuration"))
    band = base_chart.mark_area(opacity=0.1).encode(y="lower:Q", y2="upper:Q")
    line = base_chart.mark_line().encode(y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
                                      tooltip=["series:N", "progress:Q", "value:Q"])
    st.altair_chart((band + line).interactive(), width="stretch")


plot_curves("training_loss", "Training logistic loss")
plot_curves("validation_accuracy", "Validation accuracy")
st.caption("Lines interpolate recorded snapshots over the shared range of seed progress; bands show sample SD, not confidence intervals. "
           "History work counts are results received at each snapshot. Final raw totals also include post-budget draining.")

st.subheader("Worker throughput and staleness")
scaling = selected.groupby(["workers", "method_label", "scenario"], as_index=False).agg(
    throughput=("worker_throughput", "mean"), mean_staleness=("mean_staleness", "mean"),
    max_staleness=("max_staleness", "mean"))
scaling["series"] = scaling.method_label + " · " + scaling.scenario
chart = alt.Chart(scaling).mark_line(point=True).encode(
    x=alt.X("workers:O", title="OS worker processes"), y=alt.Y("throughput:Q", title="Gradient results / second"),
    color="series:N", tooltip=list(scaling.columns))
st.altair_chart(chart, width="stretch")
st.dataframe(scaling, hide_index=True, width="stretch")
st.caption("Staleness counts applied server updates since the worker snapshot. Throughput and staleness statistics use in-budget arrivals. "
           "Mini-batching changes update size, so its update-based staleness has a different scale.")

st.subheader("Experiment results")
st.dataframe(selected, width="stretch")
st.download_button("Download selected results CSV", selected.to_csv(index=False), "selected_real_worker_results.csv", "text/csv")
st.subheader("Hardware information")
st.json(hardware)
st.subheader("Main observations")
# Compute observations from the selected rows, rather than embedding benchmark numbers.
observations = selected.groupby("method_label").agg(loss=("training_loss", "mean"),
    throughput=("worker_throughput", "mean"), discarded=("gradients_stale", "sum"))
for label, row in observations.iterrows():
    st.write(f"{label}: mean final train loss {row.loss:.4f}; {row.throughput:.1f} gradients/s; "
             f"{int(row.discarded):,} stale gradients rejected across the selected runs.")
st.write("More updates or gradients do not by themselves establish better optimization. Compare all progress axes. "
         "These short runs use unequal effective batches and a fixed common learning rate. Natural scheduling, "
         "IPC, evaluation, CPU contention, and startup costs can outweigh parallel computation benefits.")
with st.expander("Generated experiment summary"):
    summary = choice / "experiment_summary.md"
    if summary.exists():
        st.markdown(summary.read_text())
