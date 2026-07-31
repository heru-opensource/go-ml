# sklexport

Python tools that bridge scikit-learn and go-ml. They are plain scripts; the
only third-party dependency is the same `scikit-learn`/`numpy` you trained with.
Use the project's virtualenv (`.venv`) or any environment that can unpickle the
model.

## Scripts

| Script | Purpose |
| --- | --- |
| `export.py` | Serialize a fitted estimator (or every supported estimator in a dict bundle) to the `go-ml/v1` JSON format that the Go package loads/compiles. |
| `make_fixtures.py` | Generate validation fixtures: inputs plus scikit-learn's exact `predict_proba`/`predict` outputs, which the Go test suite asserts against. |
| `train_examples.py` | Train go-ml's example/test models from public datasets and write both their exports and fixtures. Drives `make regen`. |

## Usage

```sh
# Export a fitted estimator (or one entry / all entries of a dict bundle):
python -m sklexport.export model.pkl -o model.json
python -m sklexport.export bundle.pkl --key clf -o clf.json
python -m sklexport.export bundle.pkl --all-keys -o ../../testdata/models/

# Generate the fixtures for a pickled estimator:
python make_fixtures.py model.pkl -o ../../testdata/fixtures/

# Or do both for the bundled example/test models in one step:
python train_examples.py --models-dir ../../testdata/models --fixtures-dir ../../testdata/fixtures
```

(Run from this directory, or add it to `PYTHONPATH`; `make_fixtures.py` and
`train_examples.py` import `export`.)

## Supported estimators

| scikit-learn estimator | Notes |
| --- | --- |
| `RandomForestClassifier` | Single-output only (`n_outputs_ == 1`). |
| `ExtraTreesClassifier` | Same payload as a random forest — the two are fitted differently but predict identically. |

Sample weighting, including `class_weight="balanced"`, needs no special
handling: scikit-learn applies it while fitting, so it is already part of the
leaf distributions that get exported.

## The `go-ml/v1` format

A JSON envelope wraps a type-specific model object, with `type` naming the
scikit-learn estimator class:

```json
{ "format": "go-ml/v1", "type": "RandomForestClassifier", "model": { ... } }
```

A forest stores each tree as flat, parallel arrays mirroring scikit-learn's
`sklearn.tree._tree.Tree` (`left`, `right`, `feature`, `threshold`,
`missing_left`) plus a flattened, L1-normalized leaf `value` matrix. Non-finite
floats — notably the `±Inf` thresholds scikit-learn uses for pure missing-value
splits — are encoded as the JSON strings `"Infinity"`, `"-Infinity"`, `"NaN"`,
because standard JSON cannot represent them and Go's decoder rejects the bare
tokens Python emits.

## Adding a new estimator type

Register an exporter in `export.py`:

```python
@register("YourEstimator")
def _export_your(est) -> dict:
    return { ... }   # the type-specific "model" object
```

Stack the decorator when several estimators serialize the same way, as
`RandomForestClassifier` and `ExtraTreesClassifier` do.

Then implement the matching decoder on the Go side and register it with
`goml.Register("YourEstimator", ...)`.
