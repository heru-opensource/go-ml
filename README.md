# go-ml

[![CI](https://github.com/heru-opensource/go-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/heru-opensource/go-ml/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/heru-opensource/go-ml.svg)](https://pkg.go.dev/github.com/heru-opensource/go-ml)

> [!CAUTION]
> **This library is fully AI-generated.** The implementation, tests, examples and
> documentation were all written by an AI agent.
>
> Every shipped model is checked against scikit-learn's own outputs in this
> repository's test suite, but its only real-world validation is against
> production workloads specific to Heru, Inc. Anything those workloads do not
> exercise — other estimator types, other hyper-parameters, other scikit-learn
> versions, other input regimes — is unproven, and *exactness against a moving
> upstream* is exactly where that gap is most likely to hurt.
>
> Use it with caution: read the code, validate your own exported model against
> your own scikit-learn outputs, and treat
> [Fidelity and its limits](#fidelity-and-its-limits) as claims to verify rather
> than as guarantees.

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

## Taste

Once, in Python:

```sh
python -m sklexport.export model.pkl -o model.json
```

Forever after, in Go:

```go
import (
    goml "github.com/heru-opensource/go-ml"
    _ "github.com/heru-opensource/go-ml/ensemble" // registers the forest models
)

// Load once at startup — or //go:embed the JSON, or compile it in with go-ml-gen.
clf, err := goml.LoadClassifierFile("model.json")
if err != nil {
    log.Fatal(err)
}

proba, _ := clf.PredictProba([][]float64{{1.2, 3.4, math.NaN()}}) // NaN = missing feature
labels, _ := clf.Predict([][]float64{{1.2, 3.4, math.NaN()}})
fmt.Println(clf.Classes(), proba, labels)
```

The same call shape as scikit-learn, none of scikit-learn's per-call overhead —
and one static binary to deploy.

## Install

```sh
go get github.com/heru-opensource/go-ml
```

Requires Go 1.26 or newer. There are no third-party dependencies: the library is
standard library only.

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

## Usage

The API mirrors scikit-learn and is the same for every classifier — program
against the `goml.Classifier` interface. `X` is a batch of samples (one inner
slice per sample); use `math.NaN` for a missing feature.

### Load a model and predict

```go
import (
    goml "github.com/heru-opensource/go-ml"
    _ "github.com/heru-opensource/go-ml/ensemble" // register the models you use
)

clf, err := goml.LoadClassifierFile("model.json")
if err != nil {
    log.Fatal(err)
}

proba, _ := clf.PredictProba([][]float64{{1.2, 3.4, math.NaN()}})
labels, _ := clf.Predict([][]float64{{1.2, 3.4, math.NaN()}})
fmt.Println(clf.Classes(), proba, labels)
```

### Build inputs by name, not by position

Feature order is part of a model. A vector assembled in the wrong order is made
of individually valid numbers, so nothing downstream can catch it — the model
predicts confidently from nonsense. When the estimator was fitted on a named
frame, scikit-learn records `feature_names_in_`, the export carries it, and
`goml.Assembler` puts values in the right columns for you:

```go
fmt.Println(clf.FeatureNames()) // [sepal_length sepal_width petal_length petal_width]

a, err := goml.NewAssembler(clf) // ErrNoFeatureNames if the export has none
if err != nil {
    log.Fatal(err)
}

row, err := a.Row(map[string]float64{ // any order; a wrong name is an error
    "petal_length": 1.4,
    "sepal_width":  3.5,
    "sepal_length": 5.1,
    // petal_width omitted → math.NaN, the missing-feature convention
})
proba, _ := clf.PredictProba([][]float64{row})
```

A retrain that reorders or renames features changes the export, so callers that
assemble by name keep working and callers that got it wrong hear about it. Names
survive static compilation too — `go-ml-gen` emits them alongside the model.

### Embed the model in your binary

```go
import _ "embed"

//go:embed model.json
var modelJSON []byte

var clf, _ = goml.LoadClassifierBytes(modelJSON)
```

### Compile the model into Go source (fastest, no runtime parsing)

```sh
go run github.com/heru-opensource/go-ml/cmd/go-ml-gen \
    -pkg models -var Model -o models/model_gen.go model.json
```

```go
import "your/module/models"

proba, _ := models.Model.PredictProba(X) // models.Model is a package-level var
```

See [Examples](#examples) for runnable programs covering all three approaches.

## Examples

Runnable programs, each narrating its own output and exiting. Between them they
cover the three ways a model reaches production — pick the one that matches your
deployment:

```sh
go run ./examples/classify   # compiled in as Go source (go-ml-gen)
go run ./examples/serve      # embedded JSON (//go:embed), served over HTTP
go run ./examples/batch      # loaded from a file, scoring a CSV in bulk
```

- [`examples/classify`](examples/classify/main.go) — two statically compiled
  models, a random forest and a balanced extra-trees model, predicted through
  one `goml.Classifier` interface. Nothing in the program depends on which
  estimator it holds.
- [`examples/serve`](examples/serve/main.go) — the service shape: the model is
  embedded in the binary, decoded once at startup, and shared by every handler
  with no pool and no lock. Shows JSON `null` as a missing feature, a wrong
  feature count answered as `400` via `errors.Is(err, goml.ErrNumFeatures)`, and
  concurrent requests agreeing exactly.
- [`examples/batch`](examples/batch/main.go) — offline scoring: a model loaded
  from a file, a CSV in and a CSV out, an empty field as a missing feature, and
  the whole file passed to a single `PredictProba` call so the rows can be spread
  across goroutines. Takes `-model`, `-csv` and `-workers`, so it doubles as a
  scoring tool for your own export.

The API documentation carries
[runnable godoc examples](https://pkg.go.dev/github.com/heru-opensource/go-ml#pkg-examples)
as well, including a whole-file one that implements and registers a new
estimator type. They run in CI, so they cannot drift from the code.

## Exporting a model

[`tools/sklexport`](tools/sklexport) serializes a fitted estimator to the
`go-ml/v1` format. The format and how to add an estimator type to it are
documented there.

```sh
python -m sklexport.export your_model.pkl -o model.json
```

## Fidelity and its limits

What "matches scikit-learn" means here, precisely. Items 1–9 are what the test
suite pins; items 10–13 are the boundaries of the claim.

1. **float32 narrowing.** scikit-learn casts `X` to `float32` before traversal,
   so every split compares `float64(float32(x)) <= threshold` with the threshold
   left at `float64`. Values within a `float32` ULP of a threshold therefore
   route identically in both languages.
2. **Missing values.** A `NaN` feature does not compare: it follows the node's
   learned `missing_go_to_left` flag. The `±Inf` thresholds scikit-learn writes
   for pure missing-value splits round-trip through the export as string
   sentinels, because JSON cannot spell them.
3. **Per-tree normalization, then the mean.** Each tree contributes the
   L1-normalized class distribution of the leaf reached; the ensemble's
   probability is the mean over trees — the arithmetic scikit-learn's
   `predict_proba` performs, in the same order.
4. **Tie-breaking.** `Predict` takes the arg-max with ties resolved toward the
   lowest column index, matching `numpy.argmax`.
5. **Weighting is fit-time.** `class_weight="balanced"` and `sample_weight`
   change the fitted leaves, not the prediction arithmetic, so a weighted model
   needs no special handling — and the shipped `extratrees_balanced` model is
   fitted that way so this is tested rather than assumed.
6. **Determinism under parallelism.** The single-goroutine and row-parallel paths
   are bit-for-bit identical to scikit-learn's single-threaded summation; the
   tree-parallel path sums partial results in a fixed order, so it is
   deterministic and agrees to floating-point rounding.
7. **Static compilation preserves everything.** Generated Go emits every float64
   in a form that parses back to identical bits, and it is validated against the
   same scikit-learn fixtures as the JSON path.
8. **Concurrency.** Loaded models are safe for concurrent use by multiple
   goroutines.
9. **Feature names travel with the model.** When scikit-learn recorded
   `feature_names_in_`, the export carries it, `Model.FeatureNames` returns it in
   column order, and static compilation preserves it — so the input contract is
   the model's to state rather than the caller's to remember. An export without
   names is ordinary: `FeatureNames` returns nil and everything else is
   unchanged.
10. **Prediction only.** No training, no `partial_fit`, no preprocessing
    pipelines. Whatever your Python code does to features before
    `predict_proba`, your Go code must do too.
11. **Single-output classifiers with numeric labels.** `n_outputs_ == 1` only,
    and class labels come across as `float64` (scikit-learn's `classes_` cast);
    string or otherwise non-numeric labels are out of scope. No regressor ships
    yet, though the interfaces and registry are already generic over them.
12. **go-ml is more permissive than scikit-learn about extreme inputs.**
    scikit-learn's `check_array` rejects `±Inf`, and any value that overflows
    `float32` (`|x| > ~3.4e38`) becomes `Inf` in that cast and is rejected too.
    go-ml validates only the feature *count* — that per-call validation is
    precisely the cost being avoided — so such inputs get an answer here and an
    exception there. Within the finite `float32` range, the two agree.
13. **Upstream is not pinned.** The fixtures in this repository were produced
    with scikit-learn 1.9. The exporter reads only the public `tree_` arrays,
    which have been stable for a long time, but nothing here can promise that a
    future scikit-learn predicts the way today's does. Re-validate when you
    upgrade.

**Validating your own model** is the same procedure the repo runs on itself:
generate reference outputs from your fitted estimator, then assert against them
from Go.

```sh
cd tools/sklexport
python export.py your_model.pkl -o /path/to/model.json
python make_fixtures.py your_model.pkl -o /path/to/fixtures/
```

The fixture holds the input rows plus scikit-learn's exact `predict_proba` and
`predict` for them, including rows with missing values and rows placed a hair
either side of real split thresholds. [`validation_test.go`](validation_test.go)
is the handful of Go that compares the two.

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

## FAQ

**Why not call Python from Go?** Because the expensive part of a scikit-learn
prediction, one sample at a time, is not the tree traversal — it is
`check_array`, dtype conversion and dispatch. Crossing a process or FFI boundary
adds to that rather than removing it. Exporting the fitted model deletes the
whole layer, along with the Python runtime in your deployment image.

**Why a JSON export instead of reading the pickle?** A pickle is executable
Python bound to the exact library versions that wrote it; it is neither safe nor
stable to read from another language. The export is a plain, versioned document
holding the fitted arrays — auditable, diffable, and readable by a Go program
that has never heard of Python.

**Why are class labels `float64`?** scikit-learn's `classes_` is numeric for the
models supported here, and one numeric type keeps `Predict` allocation-free and
the interfaces free of `any`. Label-encoding categorical targets is the normal
scikit-learn workflow, and mapping the codes back is your program's business.

**Does it train models?** No, and it is not meant to. Training belongs where the
data science happens; this library only makes a fitted model cheap to serve.

**How do I know my Go and Python agree?** Do not take it on faith — generate a
fixture from your own estimator and compare, as described in
[Fidelity and its limits](#fidelity-and-its-limits). That is exactly what CI does
for the models in this repository.

**What happens when I upgrade scikit-learn?** Re-export, regenerate the fixture,
re-run the comparison. An export captures an already-fitted tree, so an upgrade
that changes *training* cannot affect it — but one that changed *prediction*
would, and the fixture is what would catch it.

## Roadmap

Deliberately out of scope today, and documented rather than half-built:

- **Regressors** — `RandomForestRegressor` / `ExtraTreesRegressor` are the
  obvious next step: `goml.Regressor` and the registry already accommodate them,
  and a leaf holds the same shape of payload.
- **Gradient boosting** — `HistGradientBoostingClassifier` has a different tree
  representation (binned thresholds, its own missing-value handling), so it needs
  its own decoder rather than a reuse of `tree`.
- **Multi-output models** — the export format carries `n_outputs`, and the
  loaders reject anything but 1 rather than silently mis-predicting.
- **Non-numeric class labels** — would mean a label type parameter or a side
  table in the envelope; deliberately not guessed at yet.
- **Preprocessing pipelines** — scalers and encoders are a far larger surface
  than trees, and a partial implementation would be worse than none.

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
| `examples/` | Runnable examples: static compilation, an HTTP service, CSV batch scoring. |
| `internal/jsonx/` | Tolerant float decoding (handles `±Inf`/`NaN` sentinels). |
| `testdata/` | Exported models and scikit-learn reference outputs used by the tests. |

## Testing

```sh
make test      # all tests, incl. bit-exact validation against scikit-learn outputs
make race      # the same suite under the race detector, as CI runs it
make lint      # golangci-lint v2, same configuration as CI
make examples  # run every example program
make regen     # retrain models + rebuild fixtures and generated code (needs the venv)
```

The models and fixtures under `testdata/` are trained from scratch on scikit-learn's
Iris dataset and synthetic `make_classification` datasets, one of them deliberately
imbalanced — no external data — so the corpus is fully self-contained and
reproducible with `make regen`.

CI runs the suite with `-race` on Linux and macOS, repeats it to shake out
flakes, executes the godoc examples and every example program, and checks that
the committed generated artifacts are exactly what `make gen` produces today.
Python is deliberately *not* in CI: the models and fixtures are committed
artifacts, and regenerating them requires one specific scikit-learn build.

## Releasing

A Go module is published by pushing a semver tag — there is no registry to upload
to. The module proxy fetches the tag from this repository on demand, and
pkg.go.dev indexes it from there.

```sh
# 1. land the release notes FIRST — a `## [0.1.0]` heading in CHANGELOG.md is a
#    hard requirement, and the tag cannot be reused if you forget it. Push that
#    to main.
# 2. then tag the commit CI is green on:
git tag v0.1.0
git push origin v0.1.0
```

Pushing the tag triggers [`release.yml`](.github/workflows/release.yml), which
re-runs every CI gate against the tagged commit, checks the tag is one Go can
actually consume and that the version is documented, cuts a GitHub Release with
generated notes, and asks proxy.golang.org for the version so pkg.go.dev indexes
it promptly instead of waiting for someone's first `go get`.

**Tags are effectively immutable.** Once the proxy has fetched a version it caches
it permanently; moving or deleting the tag does not un-publish anything, and
consumers may already have the old bytes in their `go.sum`. A bad release is
fixed by cutting the next patch version, never by retagging. That is why the
workflow's pre-flight checks fail loudly rather than trying to paper over
anything:

| Check | Why it is fatal |
|---|---|
| strict semver (`vMAJOR.MINOR.PATCH[-pre]`) | Go silently ignores tags it cannot parse, so a typo is a release that never appears |
| major version agrees with the module path | `v2.0.0` needs the module path to end in `/v2`, or nobody can import it |
| no `replace` directives in `go.mod` | `replace` does not apply to consumers, so the module would not resolve for them |
| `## [VERSION]` heading in `CHANGELOG.md` | every released version must be documented; an undocumented release is not a release |

Because the changelog check is a hard gate, write the entry before you tag. If a
tag does fail this check, no GitHub Release is written — add the entry on the
default branch and cut the next patch version.

Everything here is one module, including the examples and the `testdata/` corpus,
so a consumer gets a library that is verifiable from the tag alone. The Python
tooling under `tools/` ships with it but is not on any Go import path.

## License

MIT — see [LICENSE](LICENSE).
