# go-ml developer tasks.
#
# `regen` retrains the example/test models from public datasets and rebuilds
# everything derived from them (JSON exports, fixtures, and statically generated
# Go) using the project's Python venv. The other targets need only the Go
# toolchain. The standalone Python benchmark has its own venv; see benchmark/.

GO  ?= go
PY  ?= .venv/bin/python

MODELS_DIR   := testdata/models
FIXTURES_DIR := testdata/fixtures
GEN_DIR      := examples/classify/models

.PHONY: all build test vet fmt bench regen train gen doc clean

all: fmt vet test

build:
	$(GO) build ./...

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

fmt:
	$(GO) fmt ./...

# Go prediction benchmark (sequential vs. parallel, single-row vs. batch).
bench:
	$(GO) run ./cmd/go-ml-bench -model $(MODELS_DIR)/forest_bench.json

# Retrain models + fixtures, then regenerate the statically compiled example.
regen: train gen

train:
	PYTHONPATH=tools/sklexport $(PY) tools/sklexport/train_examples.py \
		--models-dir $(MODELS_DIR) --fixtures-dir $(FIXTURES_DIR)

# Compile the example models into Go source for examples/classify.
gen: build
	$(GO) run ./cmd/go-ml-gen -pkg models -var Iris -o $(GEN_DIR)/iris_gen.go $(MODELS_DIR)/iris.json
	$(GO) run ./cmd/go-ml-gen -pkg models -var ExtraTreesBalanced \
		-o $(GEN_DIR)/extratrees_balanced_gen.go $(MODELS_DIR)/extratrees_balanced.json

doc:
	$(GO) doc -all ./...

clean:
	$(GO) clean ./...
