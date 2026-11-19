from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WorkerConfig:
    method: str
    workers: int = 2
    scenario: str = "natural"
    seed: int = 11
    time_budget: float = 2.0
    learning_rate: float = 0.05
    sample_batch_size: int = 128
    threshold: int | None = None
    minibatch_gradients: int = 4
    evaluation_interval: float = 0.25
    startup_timeout: float = 60.0
    task_timeout: float = 30.0
    shutdown_timeout: float = 5.0
    # Test/debug option: bound completed updates without replacing the time budget.
    max_updates: int | None = None
    worker_delays: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.method not in {"sync", "async", "filtered", "minibatch"}:
            raise ValueError("Unknown method")
        if self.scenario not in {"natural", "moderate", "severe"}:
            raise ValueError("Unknown delay scenario")
        for key in ("time_budget", "learning_rate", "evaluation_interval", "startup_timeout",
                    "task_timeout", "shutdown_timeout"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if min(self.workers, self.sample_batch_size, self.minibatch_gradients) < 1 or self.seed < 0:
            raise ValueError("Invalid count or seed")
        if self.method in {"filtered", "minibatch"}:
            if self.threshold is None or self.threshold < 0:
                raise ValueError("Filtered algorithms require a nonnegative threshold")
        elif self.threshold is not None:
            raise ValueError("Unfiltered algorithms must not specify a threshold")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be positive")
        if self.worker_delays is not None:
            if len(self.worker_delays) != self.workers or any(
                    not math.isfinite(d) or d < 0 for d in self.worker_delays):
                raise ValueError("One finite nonnegative delay is required per worker")

    def delay(self, worker_id):
        if self.worker_delays is not None:
            return self.worker_delays[worker_id]
        # At least one slow worker, including the single-worker baseline.
        slow = worker_id >= self.workers - max(1, self.workers // 4)
        return {"natural": 0.0, "moderate": 0.005, "severe": 0.025}[self.scenario] if slow else 0.0

    @property
    def method_label(self):
        if self.method == "filtered":
            return f"Filtered (tau={self.threshold})"
        if self.method == "minibatch":
            return f"Async-MB (B={self.minibatch_gradients}, tau={self.threshold})"
        return {"sync": "Synchronous", "async": "Vanilla async"}[self.method]

    @property
    def run_id(self):
        return f"{self.method}_tau{self.threshold}_w{self.workers}_{self.scenario}_s{self.seed}"

    def metadata(self):
        return dict(asdict(self), run_id=self.run_id, method_label=self.method_label,
                    effective_batch_size=self.sample_batch_size * (
                        self.workers if self.method == "sync" else
                        self.minibatch_gradients if self.method == "minibatch" else 1))
