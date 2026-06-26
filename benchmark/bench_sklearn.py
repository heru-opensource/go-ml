"""Benchmark scikit-learn's RandomForestClassifier prediction.

This is the Python half of the go-ml performance comparison. It trains the same
``forest_bench`` model the Go benchmark loads (identical dataset, parameters and
seed), then times *prediction only* — the model is fit and warmed up first, and
neither training nor model loading is included in the reported numbers. That
keeps the comparison apples-to-apples: both sides assume the model is already
loaded and measure the cost of classifying.

Prediction is timed single-threaded (``n_jobs=1``) so it lines up with go-ml's
sequential path; go-ml additionally parallelizes across goroutines, which the Go
benchmark reports separately.

See README.md in this directory for setup and how to run the Go side.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# forest_bench spec — identical to tools/sklexport/train_examples.py so this
# times the very same forest that testdata/models/forest_bench.json encodes.
SAMPLES = 2000
FEATURES = 30
INFORMATIVE = 15
CLASSES = 2
TREES = 200
MAX_DEPTH = 9
SEED = 0


def build_model():
    X, y = make_classification(
        n_samples=SAMPLES, n_features=FEATURES, n_informative=INFORMATIVE,
        n_classes=CLASSES, random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    X = X.astype(np.float64)
    X[rng.random(X.shape) < 0.05] = np.nan  # match training-time missing values
    clf = RandomForestClassifier(
        n_estimators=TREES, max_depth=MAX_DEPTH, random_state=SEED, n_jobs=1
    )
    clf.fit(X, y)
    return clf


def measure(clf, n, min_time):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, clf.n_features_in_)) * 3.0
    clf.predict_proba(X)  # warm up
    iters, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < min_time:
        clf.predict_proba(X)
        iters += 1
    dt = (time.perf_counter() - t0) / iters
    return dt


def fmt(seconds):
    if seconds < 1e-6:
        return f"{seconds*1e9:8.1f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:8.2f} us"
    return f"{seconds*1e3:8.3f} ms"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-time", type=float, default=2.0,
                   help="measurement time per case, seconds")
    args = p.parse_args(argv)

    clf = build_model()
    print(f"model: forest_bench (RandomForestClassifier, {clf.n_estimators} trees, "
          f"{clf.n_features_in_} features, {len(clf.classes_)} classes), n_jobs=1\n")
    print(f"{'case':<14}{'per-op':>12}{'per-row':>12}")
    print(f"{'----':<14}{'------':>12}{'-------':>12}")
    for n in (1, 256, 1000):
        dt = measure(clf, n, args.min_time)
        print(f"{str(n)+' rows':<14}{fmt(dt):>12}{fmt(dt/n):>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
