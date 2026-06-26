# Benchmarks

The Python half of the go-ml performance comparison. It measures scikit-learn's
`RandomForestClassifier.predict_proba` so the numbers can be set beside the
compiled Go benchmark.

Both sides time the **same model** (`forest_bench`: 200 trees, 30 features, 2
classes — the spec is duplicated here and in `tools/sklexport/train_examples.py`)
and follow the same rule: **prediction only**. The model is fit/loaded and warmed
up first; training and model-loading time are excluded, because in practice a
model is loaded once and then used to classify many times.

scikit-learn is timed single-threaded (`n_jobs=1`) to line up with go-ml's
sequential path; go-ml additionally parallelizes across goroutines, which its
benchmark reports separately.

## Setup

This directory is self-contained — create a fresh virtualenv just for it:

```sh
cd benchmark
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

(Or with [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -r requirements.txt`.)

## Run

Python (this directory):

```sh
python bench_sklearn.py            # ~10s; use --min-time to change per-case duration
```

Go (from the repository root):

```sh
go run ./cmd/go-ml-bench -model testdata/models/forest_bench.json
```

Then compare the `per-op` (single call) and `per-row` (amortized) columns. The
Go tool prints both a sequential and a parallel figure; `bench_sklearn.py` prints
the single-threaded figure.

## What you should see

Compiled Go is dramatically faster on single-sample latency (no per-call
validation overhead, no interpreter) and a few times faster on raw single-
threaded throughput, with goroutine parallelism adding more on top for batches.
The exact ratios are hardware-dependent; see the top-level README's *Performance*
section for representative figures and how to read them.
