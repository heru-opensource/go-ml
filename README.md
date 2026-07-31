# go-ml

[![Go Reference](https://pkg.go.dev/badge/github.com/heru-public/go-ml.svg)](https://pkg.go.dev/github.com/heru-public/go-ml)

Run trained [scikit-learn](https://scikit-learn.org/) models in pure Go, with
output that matches the original estimator **to floating-point rounding**
(observed max difference across the shipped models: `9.4e-16`).

A model is exported once from Python into a portable JSON document and then
either loaded at startup or — for maximum speed and zero runtime dependencies —
**compiled directly into your Go binary**. There is no Python, no pickle, and no
model file at runtime.

```
scikit-learn estimator ──(tools/sklexport)──▶ go-ml/v1 JSON ──┬──▶ goml.Load(...)         (embed & load)
                                                              └──▶ go-ml-gen ──▶ .go file (compile in)
```

The framework is generic: each estimator type registers a decoder and becomes
usable through the same `goml.Load` entry point, exactly like
`image.RegisterFormat` or `database/sql` drivers.

## Supported models

| scikit-learn estimator | Go package | Documentation |
| --- | --- | --- |
| `RandomForestClassifier` | [`ensemble`](ensemble) | [docs/randomforestclassifier.md](docs/randomforestclassifier.md) |
| `ExtraTreesClassifier` | [`ensemble`](ensemble) | [docs/extratreesclassifier.md](docs/extratreesclassifier.md) |

Class weighting (`class_weight="balanced"`, `sample_weight`) needs nothing
special: it is applied when scikit-learn fits the model and is already part of
what the export carries.

Each model's own documentation covers its usage, fidelity guarantees, tuning and
benchmarks. Anything general — loading, embedding, static compilation, the export
format — lives here.

## Why

* **Faithful.** Each model reproduces scikit-learn's prediction path exactly,
  including its internal `float32` cast of the input and its native missing-value
  (`NaN`) routing, so Go and Python agree bit-for-bit. Every model is validated
  against scikit-learn's own outputs (see [`validation_test.go`](validation_test.go)).
* **Fast.** No per-call validation overhead, no GIL, compiled traversal, and
  goroutine parallelism where it pays off. See [Performance](#performance).
* **Static.** Models live in your binary. Deployments are a single static
  executable — no Python runtime and no model file to ship or secure at runtime.
* **Generic.** Small interfaces (`Model`, `Classifier`, `Regressor`) plus a type
  registry make adding models straightforward (see [Adding a model](#adding-a-model)).

## Install

```sh
go get github.com/heru-public/go-ml
```

## Usage

The API mirrors scikit-learn and is the same for every classifier — program
against the `goml.Classifier` interface. `X` is a batch of samples (one inner
slice per sample); use `math.NaN` for a missing feature.

### Load a model and predict

```go
import (
    goml "github.com/heru-public/go-ml"
    _ "github.com/heru-public/go-ml/ensemble" // register the model types you use
)

clf, err := goml.LoadClassifierFile("model.json")
if err != nil {
    log.Fatal(err)
}

proba, _ := clf.PredictProba([][]float64{{1.2, 3.4, math.NaN()}})
labels, _ := clf.Predict([][]float64{{1.2, 3.4, math.NaN()}})
fmt.Println(clf.Classes(), proba, labels)
```

### Embed the model in your binary

```go
import _ "embed"

//go:embed model.json
var modelJSON []byte

var clf, _ = goml.LoadClassifierBytes(modelJSON)
```

### Compile the model into Go source (fastest, no runtime parsing)

```sh
go run github.com/heru-public/go-ml/cmd/go-ml-gen \
    -pkg models -var Model -o models/model_gen.go model.json
```

```go
import "your/module/models"

proba, _ := models.Model.PredictProba(X) // models.Model is a package-level var
```

The runnable [`examples/classify`](examples/classify) loads two statically
compiled models — a random forest and a balanced extra-trees model — and
predicts with both through the same interface.

## Exporting a model

[`tools/sklexport`](tools/sklexport) serializes a fitted estimator to the
`go-ml/v1` format. The format and how to add an estimator type to it are
documented there.

```sh
python -m sklexport.export your_model.pkl -o model.json
```

## Performance

For a single prediction with the model already loaded — the common online-serving
case — go-ml is typically **two to four orders of magnitude** faster than calling
scikit-learn, because it pays none of Python's per-call overhead (input
validation, dtype conversion, dispatch). On batches it is several times faster per
core and then scales further across goroutines.

Benchmarks are model-specific; see each model's documentation for measured
figures (e.g. [RandomForestClassifier](docs/randomforestclassifier.md#performance)).
The reproducible Python comparison harness lives in [`benchmark/`](benchmark);
the Go side is the `go-ml-bench` command:

```sh
go run ./cmd/go-ml-bench -model testdata/models/forest_bench.json
```

Both measure prediction only, with the model already loaded — an apples-to-apples
comparison.

## Concurrency

Loaded models are safe for concurrent use by multiple goroutines. Some models
parallelize a single prediction call internally; the details (and how to control
the worker count) are in each model's documentation.

## Adding a model

Top-level changes stay minimal by design. To add an estimator type:

1. Implement the model in its package behind `goml.Classifier` (or `Regressor`)
   and call `goml.Register("EstimatorName", decoder)` from an `init` function.
2. Add an exporter in [`tools/sklexport`](tools/sklexport).
3. Add a documentation file under [`docs/`](docs) and a row to the
   [Supported models](#supported-models) table above.

Everything general — the loader, the embed/compile workflow, the export envelope —
already works for the new type.

## Documentation

API docs are standard godoc:

```sh
go doc ./...                                              # terminal
go run golang.org/x/tools/cmd/godoc@latest -http=:6060    # browser, like pkg.go.dev
```

## Project layout

| Path | What |
| --- | --- |
| `.` (`goml`) | Interfaces (`Model`, `Classifier`, `Regressor`), the type registry, and the `Load*` entry points. |
| `tree/` | Decision-tree primitive: scikit-learn-exact traversal (`float32` cast, `NaN` routing). |
| `ensemble/` | Tree-ensemble models (`RandomForestClassifier`, `ExtraTreesClassifier`) over one shared prediction path. |
| `cmd/go-ml-gen/` | Generates Go source from an export (static compilation). |
| `cmd/go-ml-bench/` | Benchmarks prediction for any model file. |
| `tools/sklexport/` | Python exporter, model trainer, and test-fixture generator. |
| `benchmark/` | Standalone scikit-learn benchmark harness (its own venv). |
| `docs/` | Per-model documentation. |
| `examples/` | Runnable examples. |
| `internal/jsonx/` | Tolerant float decoding (handles `±Inf`/`NaN` sentinels). |
| `testdata/` | Exported models and scikit-learn reference outputs used by the tests. |

## Testing

```sh
make test     # all tests, incl. bit-exact validation against scikit-learn outputs
make regen    # retrain models + rebuild fixtures and generated code (needs the venv)
```

The models and fixtures under `testdata/` are trained from scratch on scikit-learn's
Iris dataset and synthetic `make_classification` datasets, one of them deliberately
imbalanced — no external data — so the corpus is fully self-contained and
reproducible with `make regen`.

## License

MIT — see [LICENSE](LICENSE).
