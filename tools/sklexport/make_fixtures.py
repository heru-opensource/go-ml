"""Generate validation fixtures: inputs + scikit-learn's exact outputs.

For each supported model in the bundle this writes ``<key>.fixture.json`` with a
batch of input rows and the reference ``predict_proba`` / ``predict`` outputs.
The Go test suite loads these and asserts its own results match to a tight
tolerance, proving the Go reimplementation is behaviorally identical to sklearn.

Input rows deliberately include three regimes:
  * ordinary random float64 values,
  * rows with NaNs (missing features -> exercise missing_go_to_left),
  * values placed *just* on either side of real split thresholds, to stress the
    float32 cast sklearn applies to X before tree traversal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

from export import load_pickle, _EXPORTERS


def _san(obj):
    """Recursively replace non-finite floats with JSON-safe string sentinels.

    Fixture inputs intentionally contain NaN (missing features); strict JSON has
    no NaN/inf, so we use the same sentinels the model exporter does and decode
    them with the same tolerant Go reader.
    """
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, list):
        return [_san(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _san(v) for k, v in obj.items()}
    return obj


def _threshold_boundary_rows(est, n_features, n_rows, rng):
    """Rows whose values sit a hair on each side of genuine split thresholds.

    These are the cases where sklearn's internal float32 cast of X can flip a
    branch versus a naive float64 comparison -- the toughest test of exactness.
    """
    # Collect real thresholds per feature from the first trees.
    thr_by_feat: dict[int, list[float]] = {}
    for e in est.estimators_[:50]:
        t = e.tree_
        for f, thr in zip(t.feature, t.threshold):
            # Skip leaves (f < 0) and pure missing-value splits (thr == +-inf):
            # nudging around an infinite threshold just produces inf/nan inputs.
            if f >= 0 and np.isfinite(thr):
                thr_by_feat.setdefault(int(f), []).append(float(thr))
    rows = []
    for _ in range(n_rows):
        row = rng.standard_normal(n_features) * 100.0
        for f in range(n_features):
            if f in thr_by_feat:
                thr = thr_by_feat[f][rng.integers(len(thr_by_feat[f]))]
                # Nudge by a sub-float32-ULP amount so the float32 cast decides.
                eps = rng.choice([-1e-9, -1e-8, 0.0, 1e-8, 1e-9]) * max(abs(thr), 1.0)
                row[f] = thr + eps
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def make_fixture(est, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nf = int(est.n_features_in_)

    blocks = []
    # 1) plain random
    blocks.append(rng.standard_normal((40, nf)) * 500.0)
    # 2) random with scattered NaNs
    nan_block = rng.standard_normal((40, nf)) * 500.0
    mask = rng.random((40, nf)) < 0.3
    nan_block[mask] = np.nan
    # a few all-NaN rows
    nan_block[0, :] = np.nan
    nan_block[1, :] = np.nan
    blocks.append(nan_block)
    # 3) threshold-boundary stressors
    blocks.append(_threshold_boundary_rows(est, nf, 60, rng))
    # 4) all zeros, large finite (float32-safe) magnitudes, and all-NaN
    extremes = np.zeros((4, nf))
    extremes[1, :] = 1e20
    extremes[2, :] = -1e20
    extremes[3, :] = np.nan
    blocks.append(extremes)

    X = np.vstack(blocks).astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = est.predict_proba(X)
        pred = est.predict(X)

    return {
        "n_features": nf,
        "classes": np.asarray(est.classes_, dtype=np.float64).tolist(),
        # X stored as float64; Go must reproduce sklearn's internal float32 cast.
        "X": X.tolist(),
        "predict_proba": np.asarray(proba, dtype=np.float64).tolist(),
        "predict": np.asarray(pred, dtype=np.float64).tolist(),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pickle")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=20260626)
    args = p.parse_args(argv)

    obj = load_pickle(args.pickle)
    items = obj.items() if isinstance(obj, dict) else [("model", obj)]

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for key, val in items:
        if type(val).__name__ not in _EXPORTERS:
            continue
        fx = make_fixture(val, args.seed + n)
        path = os.path.join(args.out, f"{key}.fixture.json")
        with open(path, "w") as f:
            json.dump(_san(fx), f, separators=(",", ":"), allow_nan=False)
        print(f"wrote {path} ({os.path.getsize(path):,} bytes)", file=sys.stderr)
        n += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
