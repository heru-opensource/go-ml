"""Export scikit-learn models to the generic go-ml JSON format.

This tool serializes a fitted scikit-learn estimator into a portable, versioned
JSON document that the Go ``github.com/heru-public/go-ml`` package can load (or
compile statically). It is deliberately *generic*: a small registry maps each
supported estimator class to an exporter function, so new model types can be
added without touching the Go side's envelope handling.

The JSON envelope is::

    {
      "format": "go-ml/v1",
      "type":   "<EstimatorClassName>",
      "model":  { ...type-specific... }
    }

For tree ensembles each tree is stored as flat, parallel arrays mirroring
scikit-learn's ``sklearn.tree._tree.Tree`` layout (children, feature, threshold,
missing-direction) plus a flattened, L1-normalized leaf ``value`` matrix. This
maps one-to-one onto Go slices for cache-friendly, allocation-free traversal.

Usage::

    python -m sklexport.export model.pkl -o model.json              # one estimator
    python -m sklexport.export bundle.pkl --key clf -o clf.json     # one dict entry
    python -m sklexport.export bundle.pkl --all-keys -o outdir/      # whole dict
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np

FORMAT = "go-ml/v1"

# Registry: estimator class name -> exporter callable(estimator) -> dict.
_EXPORTERS: dict[str, callable] = {}


def _floats(arr) -> list:
    """Float list that survives strict JSON: non-finite values become sentinels.

    Standard JSON has no inf/nan, and Go's encoding/json rejects the bare
    ``Infinity``/``NaN`` tokens Python emits with ``allow_nan=True``. Tree split
    thresholds *are* legitimately +-inf (sklearn's pure missing-value splits),
    so we encode non-finite values as the strings ``"Infinity"``, ``"-Infinity"``
    and ``"NaN"``. The go-ml loader decodes both numbers and these sentinels.
    Finite float64 values round-trip exactly via their shortest decimal repr.
    """
    out = []
    for v in arr:
        f = float(v)
        if f != f:
            out.append("NaN")
        elif f == float("inf"):
            out.append("Infinity")
        elif f == float("-inf"):
            out.append("-Infinity")
        else:
            out.append(f)
    return out


def register(name: str):
    def deco(fn):
        _EXPORTERS[name] = fn
        return fn

    return deco


def _export_tree(tree) -> dict:
    """Serialize one sklearn Tree into flat arrays.

    ``value`` is flattened row-major (node-major) and L1-normalized per node so
    each leaf holds a proper class-probability vector, exactly what a
    DecisionTreeClassifier's ``predict_proba`` returns for that leaf.
    """
    n = int(tree.node_count)
    value = np.asarray(tree.value, dtype=np.float64).reshape(n, -1)
    width = value.shape[1]
    # L1-normalize each node's value vector (no-op when already normalized,
    # but guarantees a probability vector regardless of sklearn version).
    norm = value.sum(axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    value = value / norm

    # missing_go_to_left lives on the structured ``nodes`` array.
    nodes = tree.__getstate__()["nodes"]
    missing_left = np.asarray(nodes["missing_go_to_left"], dtype=bool)

    return {
        "node_count": n,
        "value_width": int(width),
        "left": np.asarray(tree.children_left, dtype=np.int64).tolist(),
        "right": np.asarray(tree.children_right, dtype=np.int64).tolist(),
        "feature": np.asarray(tree.feature, dtype=np.int64).tolist(),
        "threshold": _floats(np.asarray(tree.threshold, dtype=np.float64)),
        "missing_left": missing_left.tolist(),
        "value": _floats(value.reshape(-1)),
    }


@register("RandomForestClassifier")
def _export_random_forest(est) -> dict:
    if getattr(est, "n_outputs_", 1) != 1:
        raise ValueError("only single-output forests are supported")
    classes = np.asarray(est.classes_, dtype=np.float64).tolist()
    return {
        "n_features": int(est.n_features_in_),
        "n_outputs": 1,
        "classes": classes,
        "trees": [_export_tree(e.tree_) for e in est.estimators_],
    }


def export_estimator(est) -> dict:
    name = type(est).__name__
    if name not in _EXPORTERS:
        raise ValueError(
            f"unsupported estimator {name!r}; supported: {sorted(_EXPORTERS)}"
        )
    return {"format": FORMAT, "type": name, "model": _EXPORTERS[name](est)}


def load_pickle(path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            return pickle.load(f)


def _write_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        # allow_nan=False: any non-finite must already be a string sentinel
        # (see _floats); a bare Infinity/NaN token would be a serializer bug.
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    print(f"wrote {path} ({os.path.getsize(path):,} bytes)", file=sys.stderr)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pickle", help="path to the pickled model or bundle dict")
    p.add_argument("--key", help="if the pickle is a dict bundle, export this key")
    p.add_argument(
        "--all-keys",
        action="store_true",
        help="export every supported estimator in a dict bundle to <out>/<key>.json",
    )
    p.add_argument("-o", "--out", required=True, help="output file (or dir for --all-keys)")
    args = p.parse_args(argv)

    obj = load_pickle(args.pickle)

    if args.all_keys:
        if not isinstance(obj, dict):
            print("--all-keys requires a dict bundle", file=sys.stderr)
            return 2
        n = 0
        for key, val in obj.items():
            if type(val).__name__ in _EXPORTERS:
                _write_json(export_estimator(val), os.path.join(args.out, f"{key}.json"))
                n += 1
        print(f"exported {n} model(s)", file=sys.stderr)
        return 0

    est = obj[args.key] if (args.key and isinstance(obj, dict)) else obj
    _write_json(export_estimator(est), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
