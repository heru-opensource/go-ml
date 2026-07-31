# ExtraTreesClassifier

A static, parallel Go reimplementation of scikit-learn's
[`sklearn.ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
(extremely randomized trees) prediction path. It is implemented in package
[`ensemble`](../ensemble) and registers itself with the parent `goml` package so
it can be loaded through `goml.Load`.

Models fitted with `class_weight="balanced"` are covered: the weighting is a
training-time concern that scikit-learn has already folded into the fitted
leaves, so nothing here has to be configured for it (see
[Class weighting](#class-weighting)).

## Usage

Through the generic interface:

```go
import (
    goml "github.com/heru-public/go-ml"
    _ "github.com/heru-public/go-ml/ensemble" // registers ExtraTreesClassifier
)

clf, _ := goml.LoadClassifierFile("extratrees.json")
proba, _ := clf.PredictProba([][]float64{{0.79, -2.55, 2.26, 0.23, 0.54, 0.15, 0.08, -0.68, 0.28, -2.4}})
labels, _ := clf.Predict(X)
```

Or through the concrete type, which adds single-row helpers and tuning:

```go
import "github.com/heru-public/go-ml/ensemble"

et, _ := ensemble.LoadExtraTreesClassifier(jsonBytes, ensemble.WithWorkers(4))
p, _  := et.PredictProbaRow(x)
c, _  := et.PredictRow(x)
fmt.Println(et.NTrees(), et.Classes())
```

If the estimator type is a property of the model file rather than of your code,
`ensemble.LoadForest` returns whichever forest the export holds as an
`ensemble.Forest` — the interface both this model and
[`RandomForestClassifier`](randomforestclassifier.md) satisfy:

```go
f, _ := ensemble.LoadForest(jsonBytes, ensemble.WithWorkers(4))
fmt.Println(f.Type(), f.NTrees()) // "ExtraTreesClassifier", 40
```

A missing feature is `math.NaN`; it is routed exactly as scikit-learn would
route it. The runnable [`examples/classify`](../examples/classify) predicts with
a statically compiled, balanced extra-trees model alongside a random forest.

## Relationship to RandomForestClassifier

Extra trees differ from a random forest **only in how scikit-learn grows the
trees**: split thresholds are drawn at random rather than optimized, and every
tree is fitted on the whole training set instead of a bootstrap sample. Once
fitted, scikit-learn predicts from them with the very same code — the average of
the per-tree leaf class distributions — so go-ml shares one implementation
between the two types, and they differ here only by name.

That is not an assumption: `TestExtraTreesMatchesForestArithmetic` asserts the
two Go types predict identically from identical trees, and each type is
separately validated against scikit-learn's own outputs.

## Class weighting

`class_weight="balanced"` (or an explicit weight dict, or `sample_weight`)
rescales each class's contribution **while fitting**. Its effect is entirely
contained in the resulting trees: split choices, and the class distribution
stored at each leaf. The export carries those leaf distributions, so a balanced
model is loaded and evaluated exactly like an unweighted one, and the
probabilities go-ml returns are the balanced probabilities scikit-learn returns.

The validation model shipped with the repo, `extratrees_balanced`, is fitted
that way on a deliberately imbalanced 3-class problem (70/20/10), so the
weighting is exercised end-to-end rather than assumed inert.

## Fidelity

`PredictProba` returns, for each sample, the **mean** over all trees of the class
distribution stored at the leaf the sample reaches; `Predict` returns the
`argmax` class label (ties resolved toward the lowest column index, matching
`numpy.argmax`). The result matches scikit-learn to floating-point rounding
(observed max difference on the shipped fixtures: `9.4e-16`).

Two details make the agreement exact, and both are reproduced:

* **float32 narrowing.** scikit-learn converts the input matrix `X` to `float32`
  before tree traversal, so a split compares `float64(float32(x)) <= threshold`
  (the threshold itself stays `float64`). Values sitting within a `float32` ULP of
  a threshold therefore route identically.
* **Missing values.** A `NaN` feature does not compare; it routes to the left
  child when the node's learned `missing_go_to_left` flag is set, otherwise to the
  right. Current scikit-learn learns those flags for extra trees too; an export
  from a version that predates that support simply has no flag set, and go-ml
  evaluates it just the same.

`TestAgainstSklearn` validates this on the exported model using inputs that
include missing values and values placed right on `float32` split boundaries.

**Limitation:** only single-output ensembles (`n_outputs_ == 1`) are supported.

## Concurrency

A loaded `ExtraTreesClassifier` is safe for concurrent use. A prediction call
parallelizes across goroutines once the work (`rows × trees`) is large enough to
outweigh the goroutine overhead; below that it runs on a single goroutine.

* The single-goroutine and batch (row-parallel) paths are **bit-for-bit
  identical** to scikit-learn's single-threaded summation.
* The tree-parallel path (used for a few rows over a very large ensemble) sums
  partial results in a fixed order, so it is deterministic and matches to
  floating-point rounding.

Control it with `ensemble.WithWorkers(n)`. `WithWorkers(1)` forces the exact
sequential path; the default (`0`) uses `GOMAXPROCS`.

## Performance

Prediction runs the same code as `RandomForestClassifier`, so the measured
figures and the reasoning behind them are the ones in
[RandomForestClassifier § Performance](randomforestclassifier.md#performance):
a single classification is orders of magnitude faster than calling scikit-learn,
and batches scale across goroutines. Cost per prediction is proportional to the
number of trees and their depth, and extra trees are typically grown deeper than
an equivalent random forest for the same data.

Time your own model with the same harness:

```sh
go run ./cmd/go-ml-bench -model testdata/models/extratrees_balanced.json
```

## Export format

An extra-trees ensemble is exported by [`tools/sklexport`](../tools/sklexport)
as a `go-ml/v1` envelope whose `model` object has the same shape a random
forest's does — the exporter is literally shared — with the envelope's `type`
naming the estimator:

```json
{
  "format": "go-ml/v1",
  "type": "ExtraTreesClassifier",
  "model": {
    "n_features": 10, "n_outputs": 1, "classes": [0, 1, 2],
    "trees": [ { "node_count": …, "value_width": 3,
                 "left": [...], "right": [...], "feature": [...],
                 "threshold": [...], "missing_left": [...], "value": [...] }, … ]
  }
}
```

Each tree is stored as flat, parallel arrays mirroring scikit-learn's
`sklearn.tree._tree.Tree`, plus a flattened, L1-normalized leaf `value` matrix
(node-major, width `value_width`) — the normalization is what turns weighted
class counts into the probability vector scikit-learn predicts from. Non-finite
thresholds are encoded as the JSON strings `"Infinity"` / `"-Infinity"` /
`"NaN"`; see the [exporter docs](../tools/sklexport/README.md) for the full
rationale.
