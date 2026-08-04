"""Train the tree-ensemble models used by go-ml's tests and examples.

The models are trained from scratch on scikit-learn's bundled Iris dataset and
synthetic ``make_classification`` datasets -- no downloads, no external data.
Each model is exported to the go-ml/v1 JSON format and paired with a reference
prediction fixture, so the whole test/example corpus is self-contained and
reproducible with ``make regen``.

Models produced:

  * ``iris``                - RandomForestClassifier, 4 features, 3 classes
                              (multiclass); small, for examples. Fitted on a
                              named frame, so its export carries
                              ``feature_names``.
  * ``forest_bench``        - RandomForestClassifier, 30 features, 2 classes;
                              larger, for performance tests.
  * ``iris_bundle``         - a go-ml/bundle-v1 artifact: two Iris estimators
                              (a cheap screen and a slower confirm) plus the
                              thresholds that make them a cascade, so the
                              multi-estimator format is exercised end to end.
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
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from export import export_bundle, export_estimator
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


# Tidy column names for the Iris model. scikit-learn's own names carry units
# ("sepal length (cm)"); these are what a real project's DataFrame tends to look
# like, and they double as the CSV header in examples/batch.
IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


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


def train(estimator, X, y, seed, feature_names=None, **params):
    """Fit an estimator, optionally on named columns.

    Passing feature_names fits on a DataFrame, which is the only way
    scikit-learn records ``feature_names_in_`` -- and therefore the only way the
    export can carry the feature order. It changes nothing about the fitted
    trees; iris is trained this way and the other two are not, so both paths
    stay covered.
    """
    clf = estimator(random_state=seed, **params)
    X = with_missing(X, frac=0.05, seed=seed)
    if feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)
    clf.fit(X, y)
    return clf


# iris_bundle spec: a two-stage cascade over the Iris data. The screen is cheap
# and shallow; anything it is not confident about goes to the slower confirm
# model. The two thresholds are the point of the bundle -- they are tuned
# numbers, as much a part of the artifact as the trees, and the alternative is
# for them to live in hand-written code that quietly drifts from the model.
BUNDLE_SCREEN_TREES = 15
BUNDLE_SCREEN_DEPTH = 3
BUNDLE_CONFIRM_TREES = 25
BUNDLE_CONFIRM_DEPTH = 6
BUNDLE_METADATA = {
    "screen_confidence": 0.9,   # below this, ask the confirm model
    "confirm_positive": 0.6,    # confirm's probability needed to call it positive
    "positive_class": 2.0,      # virginica
    "tuned_for": "specificity",
}


def build_bundle(X, y):
    screen = train(RandomForestClassifier, X, y, seed=0,
                   n_estimators=BUNDLE_SCREEN_TREES, max_depth=BUNDLE_SCREEN_DEPTH,
                   feature_names=IRIS_FEATURES)
    confirm = train(ExtraTreesClassifier, X, y, seed=1,
                    n_estimators=BUNDLE_CONFIRM_TREES, max_depth=BUNDLE_CONFIRM_DEPTH,
                    feature_names=IRIS_FEATURES)
    return export_bundle({"screen": screen, "confirm": confirm}, BUNDLE_METADATA)


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
        "iris": train(RandomForestClassifier, iX, iy, seed=0, n_estimators=100, max_depth=6,
                      feature_names=IRIS_FEATURES),
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

    bpath = os.path.join(args.models_dir, "iris_bundle.json")
    with open(bpath, "w") as f:
        json.dump(build_bundle(iX, iy), f, separators=(",", ":"), allow_nan=False)
    print(f"wrote {bpath} ({os.path.getsize(bpath):,} bytes, bundle of 2 models "
          f"+ {len(BUNDLE_METADATA)} metadata keys)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
