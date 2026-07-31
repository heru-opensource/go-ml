"""Train the tree-ensemble models used by go-ml's tests and examples.

The models are trained from scratch on scikit-learn's bundled Iris dataset and
synthetic ``make_classification`` datasets -- no downloads, no external data.
Each model is exported to the go-ml/v1 JSON format and paired with a reference
prediction fixture, so the whole test/example corpus is self-contained and
reproducible with ``make regen``.

Models produced:

  * ``iris``                - RandomForestClassifier, 4 features, 3 classes
                              (multiclass); small, for examples.
  * ``forest_bench``        - RandomForestClassifier, 30 features, 2 classes;
                              larger, for performance tests.
  * ``extratrees_balanced`` - ExtraTreesClassifier fitted with
                              ``class_weight="balanced"`` on a deliberately
                              imbalanced 3-class set, so the balanced
                              reweighting shows up in the leaf distributions and
                              the Go side is validated against it.

The model spec for ``forest_bench`` is shared, verbatim, with the standalone
benchmark harness (benchmark/bench_sklearn.py) so both sides time an identical
forest. A fraction of each training matrix is blanked to NaN so the trees learn
missing-value split directions and the fixtures can exercise that path (extra
trees learn them too in current scikit-learn; with a version too old for that,
drop the NaNs from the extra-trees dataset).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from export import export_estimator
from make_fixtures import _san, make_fixture

# Shared forest_bench spec (keep in sync with benchmark/bench_sklearn.py).
BENCH_SAMPLES = 2000
BENCH_FEATURES = 30
BENCH_INFORMATIVE = 15
BENCH_CLASSES = 2
BENCH_TREES = 200
BENCH_MAX_DEPTH = 9
BENCH_SEED = 0

# extratrees_balanced spec: an imbalanced 3-class problem (70/20/10), where
# class_weight="balanced" genuinely changes the fitted leaf distributions.
IMBAL_SAMPLES = 1200
IMBAL_FEATURES = 10
IMBAL_INFORMATIVE = 5
IMBAL_WEIGHTS = [0.7, 0.2, 0.1]
IMBAL_TREES = 40
IMBAL_MAX_DEPTH = 5
IMBAL_SEED = 0


def with_missing(X, frac, seed):
    rng = np.random.default_rng(seed)
    X = X.astype(np.float64).copy()
    X[rng.random(X.shape) < frac] = np.nan
    return X


def iris_dataset():
    d = load_iris()
    return d.data, d.target


def bench_dataset():
    return make_classification(
        n_samples=BENCH_SAMPLES,
        n_features=BENCH_FEATURES,
        n_informative=BENCH_INFORMATIVE,
        n_classes=BENCH_CLASSES,
        random_state=BENCH_SEED,
    )


def imbalanced_dataset():
    return make_classification(
        n_samples=IMBAL_SAMPLES,
        n_features=IMBAL_FEATURES,
        n_informative=IMBAL_INFORMATIVE,
        n_classes=len(IMBAL_WEIGHTS),
        weights=IMBAL_WEIGHTS,
        random_state=IMBAL_SEED,
    )


def train(estimator, X, y, seed, **params):
    clf = estimator(random_state=seed, **params)
    clf.fit(with_missing(X, frac=0.05, seed=seed), y)
    return clf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default="testdata/models")
    p.add_argument("--fixtures-dir", default="testdata/fixtures")
    args = p.parse_args(argv)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.fixtures_dir, exist_ok=True)

    iX, iy = iris_dataset()
    bX, by = bench_dataset()
    eX, ey = imbalanced_dataset()
    models = {
        "iris": train(RandomForestClassifier, iX, iy, seed=0, n_estimators=100, max_depth=6),
        "forest_bench": train(RandomForestClassifier, bX, by, seed=BENCH_SEED,
                              n_estimators=BENCH_TREES, max_depth=BENCH_MAX_DEPTH),
        "extratrees_balanced": train(ExtraTreesClassifier, eX, ey, seed=IMBAL_SEED,
                                     n_estimators=IMBAL_TREES, max_depth=IMBAL_MAX_DEPTH,
                                     class_weight="balanced"),
    }

    for i, (name, clf) in enumerate(models.items()):
        mpath = os.path.join(args.models_dir, f"{name}.json")
        with open(mpath, "w") as f:
            json.dump(export_estimator(clf), f, separators=(",", ":"), allow_nan=False)
        print(f"wrote {mpath} ({os.path.getsize(mpath):,} bytes, "
              f"{type(clf).__name__}, {clf.n_estimators} trees, "
              f"{clf.n_features_in_} features, {len(clf.classes_)} classes)", file=sys.stderr)

        fpath = os.path.join(args.fixtures_dir, f"{name}.fixture.json")
        with open(fpath, "w") as f:
            json.dump(_san(make_fixture(clf, seed=1000 + i)), f,
                      separators=(",", ":"), allow_nan=False)
        print(f"wrote {fpath} ({os.path.getsize(fpath):,} bytes)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
