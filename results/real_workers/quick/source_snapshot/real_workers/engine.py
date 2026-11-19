"""Spawned OS workers; the parent alone owns and updates the model.

Queues transport immutable snapshot copies and computed gradients. Training data
is read-only shared memory, not duplicated into every spawned Python process.
"""
from dataclasses import dataclass
import hashlib
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from queue import Empty
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.update_rule import UpdateRule


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _worker(worker_id, descriptors, inputs, outputs, seed, delay, l2):
    handles = []
    try:
        arrays = []
        for name, shape, dtype in descriptors:
            handle = shared_memory.SharedMemory(name=name)
            handles.append(handle)
            array = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
            array.flags.writeable = False
            arrays.append(array)
        x, y = arrays
        rng = np.random.default_rng(np.random.SeedSequence([seed, worker_id, 0]))
        with threadpool_limits(limits=1):
            outputs.put(dict(kind="ready", worker_id=worker_id, pid=os.getpid()))
            while True:
                task = inputs.get()
                if task is None:
                    break
                task_id, version, snapshot, batch_size = task
                start = time.perf_counter()
                snapshot.flags.writeable = False
                indices = rng.integers(0, len(y), size=batch_size)
                compute_start = time.perf_counter()
                xb = x[indices]
                gradient = xb.T @ (expit(xb @ snapshot) - y[indices]) / batch_size
                gradient[:-1] += l2 * snapshot[:-1]
                compute_end = time.perf_counter()
                delay_start = time.perf_counter()
                if delay:
                    time.sleep(delay)
                finish = time.perf_counter()
                outputs.put(dict(kind="gradient", worker_id=worker_id, pid=os.getpid(),
                                 task_id=task_id, snapshot_version=version, gradient=gradient,
                                 snapshot_sha256=array_hash(snapshot), gradient_sha256=array_hash(gradient),
                                 start=start, finish=finish,
                                 compute_seconds=compute_end - compute_start,
                                 injected_delay_seconds=finish - delay_start if delay else 0.0,
                                 requested_delay_seconds=delay, batch_size=batch_size))
    except BaseException:
        outputs.put(dict(kind="error", worker_id=worker_id, traceback=traceback.format_exc()))
    finally:
        for handle in handles:
            handle.close()


class WorkerPool:
    """One bounded task slot per worker, explicit readiness, bounded cleanup."""
    def __init__(self, problem, config):
        self.config = config
        self.context = mp.get_context("spawn")
        self.handles, self.processes, self.inputs = [], [], []
        self.outputs = self.context.Queue(maxsize=2 * config.workers + 1)
        self.ready = {}
        self.closed = False
        try:
            descriptors = []
            for array in (problem.x_train, problem.y_train):
                array = np.ascontiguousarray(array)
                handle = shared_memory.SharedMemory(create=True, size=array.nbytes)
                self.handles.append(handle)
                np.ndarray(array.shape, dtype=array.dtype, buffer=handle.buf)[:] = array
                descriptors.append((handle.name, array.shape, array.dtype.str))
            for worker_id in range(config.workers):
                queue = self.context.Queue(maxsize=1)
                self.inputs.append(queue)
                process = self.context.Process(target=_worker,
                    args=(worker_id, descriptors, queue, self.outputs, config.seed,
                          config.delay(worker_id), problem.l2), name=f"sgd-worker-{worker_id}")
                process.start()
                self.processes.append(process)
            deadline = time.perf_counter() + config.startup_timeout
            while len(self.ready) < config.workers:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Worker startup timed out")
                message = self.receive(min(0.1, max(0.001, deadline - time.perf_counter())))
                if message:
                    self.ready[message["worker_id"]] = message["pid"]
        except BaseException:
            self.close(force=True)
            raise

    def submit(self, worker_id, task_id, version, parameters):
        # Copy before Queue.put: multiprocessing's feeder serializes later.
        self.inputs[worker_id].put((task_id, version, parameters.copy(), self.config.sample_batch_size),
                                   timeout=1.0)

    def receive(self, timeout=0.1):
        try:
            message = self.outputs.get(timeout=timeout)
        except Empty:
            for process in self.processes:
                if process.exitcode is not None:
                    raise RuntimeError(f"Worker {process.name} exited unexpectedly: {process.exitcode}")
            return None
        if message["kind"] == "error":
            raise RuntimeError(f"Worker {message['worker_id']} failed:\n{message['traceback']}")
        return message

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        deadline = time.perf_counter() + self.config.shutdown_timeout
        for queue in self.inputs:
            if not force:
                try:
                    queue.put(None, timeout=0.1)
                except Exception:
                    force = True
        for process in self.processes:
            if force and process.is_alive():
                process.terminate()
            process.join(max(0, deadline - time.perf_counter()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join(1.0)
        for queue in [*self.inputs, self.outputs]:
            queue.cancel_join_thread()
            queue.close()
        for handle in self.handles:
            handle.close()
            handle.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(force=exc_type is not None)


@dataclass
class RealResult:
    config: object
    history: pd.DataFrame
    events: pd.DataFrame
    worker_timing: pd.DataFrame
    final_parameters: np.ndarray
    summary: dict


def run(problem, config):
    """Equal wall-clock training budgets; startup and draining reported separately.

    Evaluation runs in the parent while workers continue. Its cost remains inside
    the training clock. Results returned after stopping are drained and counted,
    but never update the model. A partial accepted group is not flushed.
    """
    overall_start = time.perf_counter()
    parameters = np.zeros(problem.dimension)
    group = (config.workers if config.method == "sync" else
             config.minibatch_gradients if config.method == "minibatch" else 1)
    rule = UpdateRule(group, config.threshold)
    events, history = [], []
    version = used = evaluation_seconds = queue_wait_seconds = dispatch_seconds = 0
    active, task_ids = {}, [0] * config.workers
    pool = WorkerPool(problem, config)
    startup_seconds = time.perf_counter() - overall_start
    initial_start = time.perf_counter()
    initial_metrics = problem.evaluate(parameters)
    initial_evaluation_seconds = time.perf_counter() - initial_start
    start = time.perf_counter()
    deadline = start + config.time_budget
    next_evaluation = start + config.evaluation_interval

    def dispatch(worker_id):
        nonlocal dispatch_seconds
        t = time.perf_counter()
        task_ids[worker_id] += 1
        active[worker_id] = (task_ids[worker_id], t)
        pool.submit(worker_id, task_ids[worker_id], version, parameters)
        dispatch_seconds += time.perf_counter() - t

    def counts(in_budget_only=False):
        counted = [e for e in events if e["in_budget"]] if in_budget_only else events
        accepted = sum(e["accepted"] for e in events)
        return dict(optimizer_updates=version, gradients_computed=len(counted),
                    gradients_accepted=accepted, gradients_discarded=len(counted) - accepted,
                    gradients_used=used, gradients_pending=len(rule.buffer),
                    training_examples_processed=len(counted) * config.sample_batch_size)

    history.append(dict(elapsed_seconds=0.0, **initial_metrics, **counts()))

    def record():
        nonlocal evaluation_seconds
        before = time.perf_counter()
        state = counts(in_budget_only=True)
        metrics = problem.evaluate(parameters)
        # x is when the snapshot was taken, not when expensive evaluation ended.
        history.append(dict(elapsed_seconds=before - start, **metrics, **state))
        evaluation_seconds += time.perf_counter() - before

    def consume(message, in_budget):
        nonlocal version, used
        arrival = time.perf_counter()
        worker_id = message["worker_id"]
        if worker_id not in active or active[worker_id][0] != message["task_id"]:
            raise RuntimeError("Duplicate or unexpected worker task result")
        _, dispatched = active.pop(worker_id)
        stale = version - message["snapshot_version"]
        if stale < 0 or not np.all(np.isfinite(message["gradient"])):
            raise FloatingPointError("Invalid gradient or snapshot version")
        accepted = in_budget and rule.accepts(stale)
        arrival_version = version
        updated = False
        if accepted:
            gradient = rule.add(message["gradient"])
            if gradient is not None:
                parameters[:] -= config.learning_rate * gradient
                version += 1
                used += group
                updated = True
                if not np.all(np.isfinite(parameters)):
                    raise FloatingPointError("Nonfinite model parameters")
        events.append({key: value for key, value in message.items()
                       if key not in {"kind", "gradient", "start", "finish"}} | dict(
            dispatched_seconds=dispatched - start, start_seconds=message["start"] - start,
            finish_seconds=message["finish"] - start, arrival_seconds=arrival - start,
            task_elapsed_seconds=message["finish"] - message["start"],
            dispatch_to_start_seconds=max(0.0, message["start"] - dispatched),
            result_delivery_seconds=max(0.0, arrival - message["finish"]),
            arrival_version=arrival_version, staleness=stale, accepted=bool(accepted),
            reason="accepted" if accepted else "stale" if in_budget else "after_budget",
            in_budget=in_budget, update_applied=updated, version_after=version))
        return worker_id, updated

    try:
        for worker_id in range(config.workers):
            dispatch(worker_id)
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            message = pool.receive(min(0.05, max(0.0001, deadline - before)))
            queue_wait_seconds += time.perf_counter() - before
            now = time.perf_counter()
            if message is not None:
                worker_id, updated = consume(message, now < deadline)
                stop_updates = config.max_updates is not None and version >= config.max_updates
                if now < deadline and not stop_updates:
                    if config.method == "sync":
                        if updated:
                            for worker_id in range(config.workers):
                                dispatch(worker_id)
                    else:
                        dispatch(worker_id)
                if stop_updates:
                    break
            if any(now - dispatched > config.task_timeout for _, dispatched in active.values()):
                raise TimeoutError("Worker task timed out")
            if now >= next_evaluation and now < deadline:
                record()
                next_evaluation = time.perf_counter() + config.evaluation_interval
        train_end = time.perf_counter()
        budget_counts = counts(in_budget_only=True)
        # Final model evaluation and draining are outside the optimization budget.
        final_metrics = problem.evaluate(parameters)
        history.append(dict(elapsed_seconds=train_end - start, **final_metrics, **budget_counts))
        drain_start = time.perf_counter()
        while active:
            if time.perf_counter() - drain_start > config.task_timeout:
                raise TimeoutError("Timed out draining final worker tasks")
            message = pool.receive(0.05)
            if message is not None:
                consume(message, False)
        drain_seconds = time.perf_counter() - drain_start
    except BaseException:
        pool.close(force=True)
        raise
    finally:
        shutdown_start = time.perf_counter()
        pool.close()
    shutdown_seconds = time.perf_counter() - shutdown_start
    event_frame = pd.DataFrame(events)
    timing_rows = []
    for worker_id, pid in pool.ready.items():
        frame = event_frame[event_frame.worker_id == worker_id]
        timing_rows.append(dict(worker_id=worker_id, pid=pid, gradients_computed=len(frame),
            training_examples_processed=len(frame) * config.sample_batch_size,
            worker_compute_seconds=float(frame.compute_seconds.sum()),
            worker_task_seconds=float(frame.task_elapsed_seconds.sum()),
            injected_delay_seconds=float(frame.injected_delay_seconds.sum()),
            dispatch_to_start_seconds=float(frame.dispatch_to_start_seconds.sum()),
            result_delivery_seconds=float(frame.result_delivery_seconds.sum()),
            gradients_in_budget=int(frame.in_budget.sum()),
            worker_throughput=float(frame.in_budget.sum()) / (train_end - start),
            exitcode=pool.processes[worker_id].exitcode))
    summary = dict(config.metadata(), **final_metrics, **counts(),
        gradients_started=sum(task_ids), gradients_in_flight=0,
        gradients_computed_in_budget=budget_counts["gradients_computed"],
        gradients_after_budget=int((~event_frame.in_budget).sum()),
        gradients_stale=int((event_frame.reason == "stale").sum()),
        wall_clock_seconds=train_end - start, startup_seconds=startup_seconds,
        initial_evaluation_seconds=initial_evaluation_seconds,
        evaluation_seconds=evaluation_seconds, drain_seconds=drain_seconds,
        shutdown_seconds=shutdown_seconds, end_to_end_seconds=time.perf_counter() - overall_start,
        mean_staleness=float(event_frame.loc[event_frame.in_budget, "staleness"].mean())
            if event_frame.in_budget.any() else 0.0,
        max_staleness=int(event_frame.loc[event_frame.in_budget, "staleness"].max())
            if event_frame.in_budget.any() else 0,
        p95_staleness=float(event_frame.loc[event_frame.in_budget, "staleness"].quantile(0.95))
            if event_frame.in_budget.any() else 0.0,
        worker_throughput=budget_counts["gradients_computed"] / (train_end - start),
        examples_per_second=budget_counts["training_examples_processed"] / (train_end - start),
        worker_compute_seconds=float(event_frame.compute_seconds.sum()),
        injected_delay_seconds=float(event_frame.injected_delay_seconds.sum()),
        dispatch_seconds=dispatch_seconds, queue_wait_seconds=queue_wait_seconds,
        server_coordination_seconds=max(0.0, train_end - start - evaluation_seconds - queue_wait_seconds),
        delivery_latency_seconds=float(event_frame.result_delivery_seconds.sum()),
        clean_shutdown=all(p.exitcode == 0 for p in pool.processes))
    result = RealResult(config, pd.DataFrame(history), event_frame, pd.DataFrame(timing_rows),
                        parameters.copy(), summary)
    audit(result)
    return result


def audit(result):
    s, e = result.summary, result.events
    checks = [
        s["gradients_computed"] == s["gradients_accepted"] + s["gradients_discarded"],
        s["gradients_accepted"] == s["gradients_used"] + s["gradients_pending"],
        s["gradients_computed"] == s["gradients_started"] == len(e),
        s["training_examples_processed"] == len(e) * result.config.sample_batch_size,
        s["gradients_computed_in_budget"] == int(e.in_budget.sum()),
        s["gradients_computed_in_budget"] + s["gradients_after_budget"] == len(e),
        s["optimizer_updates"] * s["effective_batch_size"] ==
            s["gradients_used"] * result.config.sample_batch_size,
        (e.staleness == e.arrival_version - e.snapshot_version).all(),
        (e.version_after == e.arrival_version + e.update_applied.astype(int)).all(),
        not e.duplicated(["worker_id", "task_id"]).any(), s["clean_shutdown"],
    ]
    if not all(checks):
        raise AssertionError(f"Result accounting failed: {checks}")
