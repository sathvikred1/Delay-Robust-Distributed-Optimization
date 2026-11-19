"""Real data, fixed stratified splits, and preprocessing fitted on training only."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.objective import LogisticProblem


@dataclass
class PreparedData:
    problem: LogisticProblem
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict


def prepare_arrays(x, y, name, split_seed=2027, max_samples=None, l2=0.001):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    indices = np.arange(len(y))
    if max_samples is not None and max_samples < len(y):
        indices, _ = train_test_split(indices, train_size=max_samples,
                                      stratify=y, random_state=split_seed - 1)
    train, holdout = train_test_split(indices, test_size=0.3,
                                     stratify=y[indices], random_state=split_seed)
    validation, test = train_test_split(holdout, test_size=0.5,
                                       stratify=y[holdout], random_state=split_seed + 1)
    scaler = StandardScaler().fit(x[train])

    def transform(rows):
        return np.ascontiguousarray(np.column_stack((scaler.transform(x[rows]), np.ones(len(rows)))))

    xt, xv, xe = (transform(rows) for rows in (train, validation, test))
    arrays = (xt, y[train], xv, y[validation], xe, y[test])
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    metadata = dict(name=name, source_samples=len(y), selected_samples=len(indices),
                    features=x.shape[1], train_samples=len(train),
                    validation_samples=len(validation), test_samples=len(test),
                    split_seed=split_seed, subset_seed=split_seed - 1,
                    split_ratio="70/15/15, stratified", l2=l2,
                    sha256=digest.hexdigest(), scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist(),
                    split_index_sha256={key: hashlib.sha256(rows.tobytes()).hexdigest()
                                        for key, rows in zip(("train", "validation", "test"),
                                                             (train, validation, test))},
                    positive_fraction={key: float(y[rows].mean())
                                       for key, rows in zip(("train", "validation", "test"),
                                                            (train, validation, test))})
    return PreparedData(LogisticProblem(xt, y[train], xv, y[validation], l2), xe, y[test], metadata)


def load_dataset(name="breast_cancer", cache_dir="data/cache", max_samples=None):
    if name == "breast_cancer":
        raw = load_breast_cancer()
        x, y = raw.data, raw.target
        target = "1=benign, 0=malignant (scikit-learn labels)"
        source = "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"
    elif name == "covtype":
        try:
            raw = fetch_covtype(data_home=str(Path(cache_dir).resolve()))
        except Exception as error:
            raise RuntimeError("Could not load Forest CoverType. Internet is needed on the first run; "
                               "populate --cache-dir on a connected machine. No synthetic fallback is used.") from error
        x, y = raw.data, (raw.target == 2).astype(float)
        target = "1=cover type 2 (Lodgepole Pine), 0=all other cover types; binary one-vs-rest"
        source = "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data = prepare_arrays(x, y, name, max_samples=max_samples)
    data.metadata.update(target=target, source=source)
    return data
