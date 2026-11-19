"""Hand-checkable process integration tests and independent data/accounting checks."""
from dataclasses import replace
import multiprocessing as mp
import os
import time

import numpy as np
import pytest
from scipy.special import expit

from real_workers.config import WorkerConfig
from real_workers.data import load_dataset, prepare_arrays
from real_workers.engine import WorkerPool, run
from src.objective import LogisticProblem
from src.update_rule import UpdateRule


@pytest.fixture
def problem():
    x, y = np.ones((16, 2)), np.ones(16)
    return LogisticProblem(x, y, x.copy(), y.copy(), l2=0)


def receive_gradient(pool, timeout=5):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        message = pool.receive(0.05)
        if message:
            return message
    pytest.fail("No worker gradient received")


def test_real_process_snapshot_and_gradient(problem):
    cfg = WorkerConfig("async", workers=1)
    before = {p.pid for p in mp.active_children()}
    with WorkerPool(problem, cfg) as pool:
        assert pool.ready[0] != os.getpid()
        snapshot = np.zeros(2)
        pool.submit(0, 1, 7, snapshot)
        snapshot[:] = 100  # Queue feeder must not see this later mutation.
        message = receive_gradient(pool)
        np.testing.assert_array_equal(message["gradient"], [-0.5, -0.5])
        assert message["snapshot_version"] == 7
        assert message["pid"] == pool.ready[0]
    assert not any(p.is_alive() for p in pool.processes)
    assert {p.pid for p in mp.active_children()} == before
    assert all(p.exitcode == 0 for p in pool.processes)


def test_worker_numerical_gradient():
    rng = np.random.default_rng(4)
    x = np.column_stack((rng.normal(size=(9, 3)), np.ones(9)))
    y = rng.integers(0, 2, 9).astype(float)
    problem = LogisticProblem(x, y, x, y, l2=0.03)
    cfg = WorkerConfig("async", workers=1, sample_batch_size=7, seed=22)
    w = rng.normal(size=4)
    indices = np.random.default_rng(np.random.SeedSequence([22, 0, 0])).integers(0, 9, 7)

    def loss(v):
        logits = x[indices] @ v
        return np.mean(np.logaddexp(0, logits) - y[indices] * logits) + 0.015 * (v[:-1] @ v[:-1])

    numerical = [(loss(w + d) - loss(w - d)) / 2e-6 for d in np.eye(4) * 1e-6]
    with WorkerPool(problem, cfg) as pool:
        pool.submit(0, 1, 0, w)
        message = receive_gradient(pool)
    np.testing.assert_allclose(message["gradient"], numerical, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("method,threshold", [("sync", None), ("async", None),
                                               ("filtered", 2), ("minibatch", 5)])
def test_algorithms_accounting_and_clean_shutdown(problem, method, threshold):
    cfg = WorkerConfig(method, workers=2, threshold=threshold, time_budget=1,
                       max_updates=3, worker_delays=(0.002, 0.01))
    result = run(problem, cfg)
    s, e = result.summary, result.events
    assert s["optimizer_updates"] == 3
    assert s["gradients_computed"] == s["gradients_accepted"] + s["gradients_discarded"]
    assert s["gradients_accepted"] == s["gradients_used"] + s["gradients_pending"]
    assert s["clean_shutdown"]
    assert len(result.worker_timing.pid.unique()) == 2
    assert (e.staleness == e.arrival_version - e.snapshot_version).all()
    if method == "sync":
        assert e.loc[e.accepted, "staleness"].eq(0).all()
        assert s["gradients_used"] == 6
    if method == "minibatch":
        assert s["gradients_used"] == 12


def test_sync_waits_for_slowest_and_averages(problem):
    result = run(problem, WorkerConfig("sync", workers=2, time_budget=2, max_updates=1,
                 learning_rate=0.1, worker_delays=(0.0, 0.08)))
    np.testing.assert_allclose(result.final_parameters, [0.05, 0.05])
    events = result.events[result.events.accepted]
    assert len(events) == 2
    assert events.iloc[-1].arrival_seconds >= 0.08
    assert not events.iloc[0].update_applied
    assert events.iloc[-1].update_applied
    assert events.snapshot_version.eq(0).all()


def test_async_uses_original_gradients_and_advances_before_slow_worker(problem):
    cfg = WorkerConfig("async", workers=2, time_budget=0.2,
                       learning_rate=0.1, worker_delays=(0.005, 0.04))
    result = run(problem, cfg)
    # Independently replay gradients for each observed snapshot; never use arrival w.
    snapshots = {0: np.zeros(2)}
    w = np.zeros(2)
    for event in result.events.itertuples():
        if event.accepted:
            g = np.full(2, expit(snapshots[event.snapshot_version].sum()) - 1)
            w -= 0.1 * g
            snapshots[event.version_after] = w.copy()
    np.testing.assert_allclose(result.final_parameters, w, atol=1e-12)
    slow = result.events[(result.events.worker_id == 1) & result.events.in_budget]
    assert (slow.staleness > 0).any()
    assert result.events.iloc[0].worker_id == 0


def test_filter_actually_discards_stale_gradients(problem):
    result = run(problem, WorkerConfig("filtered", threshold=2, workers=2, time_budget=0.25,
                                      worker_delays=(0.001, 0.05)))
    stale = result.events[result.events.reason == "stale"]
    assert len(stale) > 0
    assert (stale.staleness > 2).all()
    assert result.events.loc[result.events.accepted, "staleness"].le(2).all()


def test_inclusive_threshold_and_fixed_minibatch():
    rule = UpdateRule(4, threshold=2)
    assert rule.accepts(2) and not rule.accepts(3)
    for value in (1, 3, 5):
        assert rule.add(np.array([value], dtype=float)) is None
    np.testing.assert_array_equal(rule.add(np.array([7.])), [4.])
    assert rule.buffer == []


def test_incomplete_minibatch_is_not_flushed(problem):
    result = run(problem, WorkerConfig("minibatch", workers=1, threshold=5,
                 minibatch_gradients=100, time_budget=0.04, worker_delays=(0.01,)))
    assert result.summary["optimizer_updates"] == 0
    assert 0 < result.summary["gradients_pending"] < 100
    np.testing.assert_array_equal(result.final_parameters, np.zeros(2))
    assert result.summary["gradients_after_budget"] == 1


def test_data_reproducibility_no_leakage_and_disjoint_splits():
    x = np.arange(300, dtype=float).reshape(100, 3)
    y = np.arange(100) % 2
    a, b = prepare_arrays(x, y, "fixture"), prepare_arrays(x, y, "fixture")
    assert a.metadata["sha256"] == b.metadata["sha256"]
    assert (a.metadata["train_samples"], a.metadata["validation_samples"], a.metadata["test_samples"]) == (70, 15, 15)
    np.testing.assert_allclose(a.problem.x_train[:, :-1].mean(axis=0), 0, atol=1e-14)
    # Scaler mean must correspond to training, not the complete original data.
    assert not np.allclose(a.metadata["scaler_mean"], x.mean(axis=0))
    sets = [set(array[:, 0]) for array in (a.problem.x_train, a.problem.x_validation, a.x_test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    real = load_dataset()
    assert real.metadata["source_samples"] == 569
    assert real.problem.dimension == 31


def test_task_timeout_cleans_up(problem):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(TimeoutError, match="timed out"):
        run(problem, WorkerConfig("async", workers=1, time_budget=0.2,
                                  task_timeout=0.02, worker_delays=(0.4,)))
    assert {p.pid for p in mp.active_children()} == before


def test_worker_exception_is_propagated_and_pool_can_close(problem):
    pool = WorkerPool(problem, WorkerConfig("async", workers=1))
    try:
        pool.submit(0, 1, 0, np.zeros(99))  # Invalid matrix dimension inside child.
        with pytest.raises(RuntimeError, match="failed"):
            receive_gradient(pool)
    finally:
        pool.close(force=True)
    assert not any(p.is_alive() for p in pool.processes)


def test_dead_worker_is_detected(problem):
    pool = WorkerPool(problem, WorkerConfig("async", workers=1))
    pool.processes[0].terminate()
    pool.processes[0].join(1)
    try:
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            pool.receive(0.1)
    finally:
        pool.close(force=True)


@pytest.mark.parametrize("changes", [dict(threshold=2), dict(time_budget=float("nan")),
                                       dict(workers=0), dict(worker_delays=(float("inf"),))])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        replace(WorkerConfig("async", workers=1), **changes)


def test_result_crossing_deadline_is_excluded_from_throughput(problem, monkeypatch):
    original = WorkerPool.receive

    def delayed_receipt(self, timeout=0.1):
        message = original(self, timeout)
        if message and message["kind"] == "gradient":
            time.sleep(0.03)
        return message

    monkeypatch.setattr(WorkerPool, "receive", delayed_receipt)
    result = run(problem, WorkerConfig("async", workers=1, time_budget=0.02))
    assert result.summary["gradients_computed"] == 1
    assert result.summary["gradients_computed_in_budget"] == 0
    assert result.summary["gradients_after_budget"] == 1
    assert result.summary["worker_throughput"] == 0
    assert result.history.iloc[-1].gradients_computed == 0
