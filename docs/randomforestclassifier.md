# RandomForestClassifier

A static, parallel Go reimplementation of scikit-learn's
[`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
prediction path. It is implemented in package
[`ensemble`](../ensemble) and registers itself with the parent `goml` package so
it can be loaded through `goml.Load`.

[`ExtraTreesClassifier`](extratreesclassifier.md) shares this prediction path
and, in go-ml, this implementation; everything below applies to it unchanged.
`ensemble.LoadForest` loads either type from its export.

## Usage

Through the generic interface:

```go
import (
    goml "github.com/heru-opensource/go-ml"
    _ "github.com/heru-opensource/go-ml/ensemble" // registers the forest models
)

clf, _ := goml.LoadClassifierFile("forest.json")
proba, _ := clf.PredictProba([][]float64{{5.1, 3.5, 1.4, 0.2}})
labels, _ := clf.Predict([][]float64{{5.1, 3.5, 1.4, 0.2}})
```

Or through the concrete type, which adds single-row helpers and tuning:

```go
import "github.com/heru-opensource/go-ml/ensemble"

rf, _ := ensemble.LoadRandomForestClassifier(jsonBytes, ensemble.WithWorkers(4))
p, _  := rf.PredictProbaRow([]float64{5.1, 3.5, 1.4, 0.2})
c, _  := rf.PredictRow([]float64{5.1, 3.5, 1.4, 0.2})
fmt.Println(rf.NTrees(), rf.Classes())
```

A missing feature is `math.NaN`; it is routed exactly as scikit-learn would
route it.

## Fidelity

`PredictProba` returns, for each sample, the **mean** over all trees of the class
distribution stored at the leaf the sample reaches; `Predict` returns the
`argmax` class label (ties resolved toward the lowest column index, matching
`numpy.argmax`). The result matches scikit-learn to floating-point rounding.

Two details make the agreement exact, and both are reproduced:

* **float32 narrowing.** scikit-learn converts the input matrix `X` to `float32`
  before tree traversal, so a split compares `float64(float32(x)) <= threshold`
  (the threshold itself stays `float64`). Values sitting within a `float32` ULP of
  a threshold therefore route identically.
* **Missing values.** A `NaN` feature does not compare; it routes to the left
  child when the node's learned `missing_go_to_left` flag is set, otherwise to the
  right. scikit-learn also uses an infinite threshold for pure missing-value
  splits, which the export format and Go traversal handle.

`TestAgainstSklearn` validates this on the exported models using inputs that
include missing values and values placed right on `float32` split boundaries.

**Limitation:** only single-output forests (`n_outputs_ == 1`) are supported.

## Concurrency

A loaded `RandomForestClassifier` is safe for concurrent use. A prediction call
parallelizes across goroutines once the work (`rows × trees`) is large enough to
outweigh the goroutine overhead; below that it runs on a single goroutine.

* The single-goroutine and batch (row-parallel) paths are **bit-for-bit
  identical** to scikit-learn's single-threaded summation.
* The tree-parallel path (used for a few rows over a very large forest) sums
  partial results in a fixed order, so it is deterministic and matches to
  floating-point rounding.

Control it with `ensemble.WithWorkers(n)`. `WithWorkers(1)` forces the exact
sequential path; the default (`0`) uses `GOMAXPROCS`.

## Performance

Measured on `forest_bench` (200 trees, 30 features, 2 classes), prediction only
with the model already loaded, on a 36-core Intel i9-10980XE. scikit-learn is
timed single-threaded (`n_jobs=1`); reproduce with the harness in
[`benchmark/`](../benchmark) and `go run ./cmd/go-ml-bench`.

| Workload | scikit-learn (`n_jobs=1`) | go-ml (1 goroutine) | go-ml (parallel) |
| --- | --- | --- | --- |
| **1 sample** (latency) | 11.1 ms | **9.6 µs** | **9.3 µs** |
| 256 samples (per row) | 66.5 µs | 20.5 µs | **3.2 µs** |
| 1000 samples (per row) | 29.3 µs | 20.4 µs | **1.8 µs** |

Reading the numbers:

* **A single classification is ~1,000× faster** (9.6 µs vs 11.1 ms). Classifying
  one sample in scikit-learn is dominated by fixed per-call overhead — input
  validation and dispatch — which a compiled binary simply does not have. This is
  the realistic cost when a model is loaded once and then used to classify.
* **Raw single-threaded throughput is ~1.4–3.2× faster** per core (go-ml with one
  goroutine vs scikit-learn's Cython+NumPy), the gap widening on smaller batches
  where per-call overhead weighs more.
* **Goroutines add ~6–11× on top** for batches on this machine (256 rows:
  20.5 → 3.2 µs/row; 1000 rows: 20.4 → 1.8 µs/row).

## Feature names

If the estimator was fitted on a named frame, scikit-learn recorded
`feature_names_in_` and the export carries it, in column order:

```go
fmt.Println(clf.FeatureNames())      // nil when the export has none
a, _ := goml.NewAssembler(clf)       // build inputs by name instead of position
row, err := a.Row(map[string]float64{"petal_length": 1.4, "sepal_width": 3.5})
```

Order is part of the model, and a vector built in the wrong order is made of
individually valid numbers — no validation can catch it. Names move that
contract into the artifact, so a retrain that reorders or renames features
updates the export rather than silently breaking callers. `ensemble.WithFeatureNames`
attaches them when constructing a model by hand, which is what `go-ml-gen` emits.

## Export format

A forest is exported by [`tools/sklexport`](../tools/sklexport) as a `go-ml/v1`
envelope whose `model` object is:

```json
{
  "n_features": 30, "n_outputs": 1, "classes": [0, 1],
  "feature_names": ["age", "pressure", …],
  "trees": [ { "node_count": …, "value_width": 2,
               "left": [...], "right": [...], "feature": [...],
               "threshold": [...], "missing_left": [...], "value": [...] }, … ]
}
```

Each tree is stored as flat, parallel arrays mirroring scikit-learn's
`sklearn.tree._tree.Tree`, plus a flattened, L1-normalized leaf `value` matrix
(node-major, width `value_width`). Non-finite thresholds are encoded as the JSON
strings `"Infinity"` / `"-Infinity"` / `"NaN"`; see the
[exporter docs](../tools/sklexport/README.md) for the full rationale.
