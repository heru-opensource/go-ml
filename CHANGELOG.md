# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0, a
breaking change to the public API requires a minor bump and an entry here.

## [Unreleased]

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

[Unreleased]: https://github.com/heru-opensource/go-ml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/heru-opensource/go-ml/releases/tag/v0.1.0
