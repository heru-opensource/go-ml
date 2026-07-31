package ensemble

import (
	"encoding/json"
	"fmt"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/tree"
)

// TypeExtraTreesClassifier is the export "type" value for an extremely
// randomized trees classifier, and the value returned by
// [ExtraTreesClassifier.Type].
const TypeExtraTreesClassifier = "ExtraTreesClassifier"

func init() {
	goml.Register(TypeExtraTreesClassifier, func(raw json.RawMessage) (goml.Model, error) {
		return decodeExtraTrees(raw)
	})
}

// ExtraTreesClassifier reproduces scikit-learn's ExtraTreesClassifier
// (extremely randomized trees) prediction: each tree yields the class
// distribution of the leaf a sample reaches, and the ensemble's probability is
// the mean of those distributions over all trees (see
// [ExtraTreesClassifier.PredictProba]). Predictions are computed concurrently
// across goroutines for large batches or large ensembles.
//
// The prediction arithmetic is exactly that of [RandomForestClassifier]: the
// two estimators differ only in how scikit-learn grows their trees — extra
// trees draw split thresholds at random and, by default, fit every tree on the
// whole training set rather than on a bootstrap sample. That is a training-time
// difference, already baked into the exported trees.
//
// The same is true of class weighting: a model fitted with
// class_weight="balanced" (or any other weighting) carries the reweighted class
// distributions in its leaves, so it is loaded and evaluated like any other
// export, with no extra configuration here.
//
// An ExtraTreesClassifier is safe for concurrent use by multiple goroutines.
type ExtraTreesClassifier struct {
	forest
}

var _ Forest = (*ExtraTreesClassifier)(nil)

// NewExtraTreesClassifier constructs an ensemble from its parts. classes are
// the labels in column order; each tree's Value vectors must already be L1-
// normalized leaf class distributions of width len(classes). This is the
// constructor that statically generated code (see cmd/go-ml-gen) calls; most
// callers load a model instead with [LoadExtraTreesClassifier] or goml.Load.
func NewExtraTreesClassifier(nFeatures int, classes []float64, trees []*tree.Tree, opts ...Option) (*ExtraTreesClassifier, error) {
	f, err := newForest(nFeatures, classes, trees, opts)
	if err != nil {
		return nil, err
	}
	return &ExtraTreesClassifier{f}, nil
}

// Type returns TypeExtraTreesClassifier.
func (et *ExtraTreesClassifier) Type() string { return TypeExtraTreesClassifier }

func decodeExtraTrees(raw json.RawMessage) (*ExtraTreesClassifier, error) {
	f, err := decodeForest(raw)
	if err != nil {
		return nil, err
	}
	return &ExtraTreesClassifier{f}, nil
}

// LoadExtraTreesClassifier decodes an ensemble from a go-ml/v1 export envelope
// read from data, applying the given options. Unlike goml.Load it returns the
// concrete type, and it fails if the export holds a different estimator.
func LoadExtraTreesClassifier(data []byte, opts ...Option) (*ExtraTreesClassifier, error) {
	f, err := LoadForest(data, opts...)
	if err != nil {
		return nil, err
	}
	et, ok := f.(*ExtraTreesClassifier)
	if !ok {
		return nil, fmt.Errorf("ensemble: export is a %s, not an ExtraTreesClassifier", f.Type())
	}
	return et, nil
}
