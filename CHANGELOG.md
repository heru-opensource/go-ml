# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0, a
breaking change to the public API requires a minor bump and an entry here.

## [Unreleased]

### Added

- **Feature names travel with the model.** `tools/sklexport` writes
  `feature_names` whenever the estimator recorded `feature_names_in_`,
  `Model.FeatureNames` returns them in column order, `ensemble.WithFeatureNames`
  attaches them to a hand-built model, and `go-ml-gen` compiles them in. An
  export without names is unchanged in every respect: `FeatureNames` returns nil.
- **`goml.Assembler`** builds feature vectors from a name-keyed map, in the
  model's own column order. An unrecognised name is an error
  (`ErrUnknownFeature`); a name the model expects but the caller omits is a
  missing feature (`NaN`). This is the point of carrying names: a vector
  assembled in the wrong order consists of individually valid numbers, so it
  cannot be detected downstream — only prevented here.
- The `serve` example accepts name-keyed `"rows"` alongside positional
  `"samples"`, and the `batch` example matches the CSV header against the model's
  feature names, reordering columns rather than trusting their order.

### Changed

- **Breaking for implementers of `goml.Model`:** the interface gained
  `FeatureNames() []string`. Every model in this repository implements it, and
  callers of the interface are unaffected — but an out-of-tree `Model` needs the
  method added (return nil to keep today's behaviour).
- `testdata/models/iris.json` is now fitted on a named frame, so it carries
  feature names. The trees are byte-identical to 0.1.1's; the only difference in
  the export is the new field.

## [0.1.1] — 2026-07-31

Documentation and examples only: no change to any importable API, so upgrading
from 0.1.0 is a no-op for compiled code. Tagged so the new godoc examples reach
pkg.go.dev, which renders the latest released version.

### Added

- **Two more runnable examples.** `examples/serve` is the service shape: the
  model is embedded with `//go:embed`, decoded once at startup and shared by
  every handler, with JSON `null` as a missing feature and a wrong feature count
  answered as a `400`. `examples/batch` is offline scoring: a model loaded from a
  file, CSV in and out, an empty field as a missing feature, and the whole file
  passed to one `PredictProba` call. Both run in CI.
- **Godoc examples.** The root package now documents loading, missing features
  and the sentinel errors by example, plus a whole-file example that implements
  and registers a new estimator type. The `tree` package documents `Apply`,
  `Decide`, `AddTo` and decoding the serialized form (including its `"Infinity"`
  threshold sentinel), and `ensemble.WithWorkers` shows the worker knob.

## [0.1.0] — 2026-07-31

First release. The public API is the v0 contract. Requires Go 1.26 or newer, with
no third-party dependencies.

### Added

- **Core.** `Model`, `Classifier` and `Regressor` interfaces; a decoder registry
  (`Register`, `RegisteredTypes`) keyed by scikit-learn estimator name; and the
  `Load`, `LoadBytes`, `LoadFile` entry points plus their `LoadClassifier*`
  variants. Sentinel errors: `ErrFormat`, `ErrUnknownType`, `ErrNotClassifier`,
  `ErrNumFeatures`.
- **Trees.** Package `tree`: scikit-learn's flat-array `Tree` layout with
  traversal that reproduces `_apply_dense` exactly — the `float32` narrowing of
  input features and `missing_go_to_left` routing for `NaN` included.
- **Ensembles.** Package `ensemble`: `RandomForestClassifier` and
  `ExtraTreesClassifier` over one shared prediction path, the `Forest` interface
  and type-agnostic `LoadForest`, per-model constructors and loaders, and
  `WithWorkers` for goroutine parallelism (row-parallel for batches,
  tree-parallel for single rows over large ensembles).
- **Static compilation.** `cmd/go-ml-gen` turns an export into gofmt-clean Go
  source that builds the model from literal data, so a binary carries its model
  with no file and no runtime parsing.
- **Benchmarking.** `cmd/go-ml-bench` times single-sample latency and batch
  throughput for any model file, sequential and parallel, and `benchmark/` holds
  the matching single-threaded scikit-learn harness.
- **Python tooling.** `tools/sklexport`: the `go-ml/v1` exporter (one registry
  entry per estimator), a fixture generator that records scikit-learn's exact
  outputs, and the trainer that produces the repository's own models.
- **Validation corpus.** Models and reference fixtures under `testdata/`, trained
  from scratch on Iris and synthetic `make_classification` data — including an
  `ExtraTreesClassifier` fitted with `class_weight="balanced"` on a deliberately
  imbalanced 3-class set. Reproducible with `make regen`.
- **Docs.** Package documentation, godoc examples that run in CI, per-model pages
  under `docs/`, and a runnable `examples/classify` program that predicts with
  two statically compiled models through one interface.

### Notes on fidelity

Observed maximum `predict_proba` difference against scikit-learn across the
shipped models is `9.4e-16`, with no label mismatches, over fixtures that include
missing values and inputs placed within a `float32` ULP of real split thresholds.

Two boundaries are worth stating explicitly, since either could reasonably have
been decided the other way:

- go-ml validates only the **feature count** of an input, not its finiteness.
  scikit-learn's `check_array` rejects `±Inf` and values that overflow `float32`;
  go-ml routes them by comparison instead of erroring. Skipping per-call
  validation is where much of the speed comes from, so this is deliberate — but
  it means the two disagree outside the finite `float32` range.
- Class weighting (`class_weight="balanced"`, `sample_weight`) is treated as a
  fit-time concern with no runtime surface, because scikit-learn has already
  folded it into the exported leaf distributions.

[Unreleased]: https://github.com/heru-opensource/go-ml/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/heru-opensource/go-ml/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/heru-opensource/go-ml/releases/tag/v0.1.0
