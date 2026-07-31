package ensemble

import (
	"encoding/json"
	"fmt"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/tree"
)

// TypeRandomForestClassifier is the export "type" value for a random forest
// classifier, and the value returned by [RandomForestClassifier.Type].
const TypeRandomForestClassifier = "RandomForestClassifier"

func init() {
	goml.Register(TypeRandomForestClassifier, func(raw json.RawMessage) (goml.Model, error) {
		return decodeRandomForest(raw)
	})
}

// RandomForestClassifier reproduces scikit-learn's RandomForestClassifier
// prediction: each decision tree yields the class distribution of the leaf a
// sample reaches, and the forest's probability is the mean of those
// distributions over all trees (see [RandomForestClassifier.PredictProba]).
// Predictions are computed concurrently across goroutines for large batches or
// large forests.
//
// A RandomForestClassifier is safe for concurrent use by multiple goroutines.
type RandomForestClassifier struct {
	forest
}

var _ Forest = (*RandomForestClassifier)(nil)

// NewRandomForestClassifier constructs a forest from its parts. classes are the
// labels in column order; each tree's Value vectors must already be L1-
// normalized leaf class distributions of width len(classes). This is the
// constructor that statically generated code (see cmd/go-ml-gen) calls; most
// callers load a model instead with [LoadRandomForestClassifier] or goml.Load.
func NewRandomForestClassifier(nFeatures int, classes []float64, trees []*tree.Tree, opts ...Option) (*RandomForestClassifier, error) {
	f, err := newForest(nFeatures, classes, trees, opts)
	if err != nil {
		return nil, err
	}
	return &RandomForestClassifier{f}, nil
}

// Type returns TypeRandomForestClassifier.
func (rf *RandomForestClassifier) Type() string { return TypeRandomForestClassifier }

func decodeRandomForest(raw json.RawMessage) (*RandomForestClassifier, error) {
	f, err := decodeForest(raw)
	if err != nil {
		return nil, err
	}
	return &RandomForestClassifier{f}, nil
}

// LoadRandomForestClassifier decodes a forest from a go-ml/v1 export envelope
// read from data, applying the given options. Unlike goml.Load it returns the
// concrete type, and it fails if the export holds a different estimator.
func LoadRandomForestClassifier(data []byte, opts ...Option) (*RandomForestClassifier, error) {
	f, err := LoadForest(data, opts...)
	if err != nil {
		return nil, err
	}
	rf, ok := f.(*RandomForestClassifier)
	if !ok {
		return nil, fmt.Errorf("ensemble: export is a %s, not a RandomForestClassifier", f.Type())
	}
	return rf, nil
}
