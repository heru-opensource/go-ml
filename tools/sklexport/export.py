"""Export scikit-learn models to the generic go-ml JSON format.

This tool serializes a fitted scikit-learn estimator into a portable, versioned
JSON document that the Go ``github.com/heru-opensource/go-ml`` package can load
(or compile statically). It is deliberately *generic*: a small registry maps
each supported estimator class to an exporter function, so new model types can
be added without touching the Go side's envelope handling.

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
    python -m sklexport.export bundle.pkl --bundle --meta threshold=0.83 \\
        -o bundle.json                                               # one artifact
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
BUNDLE_FORMAT = "go-ml/bundle-v1"

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


def _feature_names(est) -> list[str] | None:
    """The estimator's input feature names, in column order, if it knows them.

    scikit-learn sets ``feature_names_in_`` only when the estimator was fitted
    on something with named columns (a DataFrame), so this is often absent --
    which is fine. When it is there it belongs in the export: the order of a
    feature vector is part of the model, and a caller that assembles it by name
    cannot silently misfeed a retrained model whose columns moved.
    """
    names = getattr(est, "feature_names_in_", None)
    if names is None:
        return None
    return [str(n) for n in names]


@register("RandomForestClassifier")
@register("ExtraTreesClassifier")
def _export_forest(est) -> dict:
    """Serialize a tree-ensemble classifier: a forest of fitted trees.

    RandomForestClassifier and ExtraTreesClassifier share this exporter. They
    grow their trees differently (optimal vs. random split thresholds, bootstrap
    samples vs. the whole training set), but once fitted they predict
    identically -- by averaging the per-tree leaf distributions -- so the same
    payload serves both. Sample weighting, including
    ``class_weight="balanced"``, is likewise a training-time concern: it is
    already folded into the leaf ``value`` vectors written out here.
    """
    if getattr(est, "n_outputs_", 1) != 1:
        raise ValueError("only single-output forests are supported")
    classes = np.asarray(est.classes_, dtype=np.float64).tolist()
    model = {
        "n_features": int(est.n_features_in_),
        "n_outputs": 1,
        "classes": classes,
        "trees": [_export_tree(e.tree_) for e in est.estimators_],
    }
    names = _feature_names(est)
    if names is not None:
        model["feature_names"] = names
    return model


def export_estimator(est) -> dict:
    name = type(est).__name__
    if name not in _EXPORTERS:
        raise ValueError(
            f"unsupported estimator {name!r}; supported: {sorted(_EXPORTERS)}"
        )
    return {"format": FORMAT, "type": name, "model": _EXPORTERS[name](est)}


def export_bundle(estimators: dict, metadata: dict | None = None) -> dict:
    """Serialize several named estimators, plus scalar metadata, as one artifact.

    A deployed model is often not one estimator: a decision may take two or
    three of them and the thresholds they were tuned against. Those thresholds
    are as much a fitted parameter as any split in a tree, and keeping them in
    hand-written code beside the model is what goes stale -- the numbers and the
    trees get updated by different hands, and nothing fails loudly when they
    disagree. A bundle ships them as one versioned file.

    ``metadata`` values may be any JSON-serializable value; go-ml stores them and
    hands them back typed, and does not interpret them.
    """
    if not estimators:
        raise ValueError("a bundle needs at least one estimator")
    models = {}
    for name, est in estimators.items():
        cls = type(est).__name__
        if cls not in _EXPORTERS:
            raise ValueError(
                f"unsupported estimator {cls!r} for bundle key {name!r}; "
                f"supported: {sorted(_EXPORTERS)}"
            )
        models[name] = {"type": cls, "model": _EXPORTERS[cls](est)}
    return {
        "format": BUNDLE_FORMAT,
        "models": models,
        "metadata": dict(metadata or {}),
    }


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


def _parse_meta(pairs: list) -> dict:
    """Parse KEY=VALUE metadata arguments, VALUE being JSON where it parses.

    So ``--meta threshold=0.83`` stores a number and ``--meta stage=screen``
    stores the string "screen", which is what either looks like it should mean.
    """
    meta = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--meta {pair!r} is not KEY=VALUE")
        try:
            meta[key] = json.loads(value)
        except json.JSONDecodeError:
            meta[key] = value
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pickle", help="path to the pickled model or bundle dict")
    p.add_argument("--key", help="if the pickle is a dict bundle, export this key")
    p.add_argument(
        "--all-keys",
        action="store_true",
        help="export every supported estimator in a dict bundle to <out>/<key>.json",
    )
    p.add_argument(
        "--bundle",
        action="store_true",
        help="write ONE go-ml/bundle-v1 file holding every supported estimator "
             "in a dict bundle, plus any --meta values",
    )
    p.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="metadata entry for --bundle, e.g. --meta threshold=0.83 "
             "--meta note='\"tuned for specificity\"' (repeatable; the value is "
             "parsed as JSON, falling back to a plain string)",
    )
    p.add_argument("-o", "--out", required=True, help="output file (or dir for --all-keys)")
    args = p.parse_args(argv)

    obj = load_pickle(args.pickle)

    if args.bundle:
        if not isinstance(obj, dict):
            print("--bundle requires a dict of {name: estimator}", file=sys.stderr)
            return 2
        estimators = {k: v for k, v in obj.items() if type(v).__name__ in _EXPORTERS}
        if not estimators:
            print("--bundle found no supported estimators in the pickle", file=sys.stderr)
            return 2
        _write_json(export_bundle(estimators, _parse_meta(args.meta)), args.out)
        print(f"bundled {len(estimators)} model(s): {', '.join(sorted(estimators))}",
              file=sys.stderr)
        return 0

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
