package goml_test

// This file is one whole-file example: how a new scikit-learn estimator type
// plugs into go-ml. It implements sklearn.dummy.DummyClassifier's "prior"
// strategy — the baseline that ignores its input and always returns the class
// distribution it was fitted on — because that keeps the prediction path down
// to a few lines. A real model does more arithmetic, and nothing else changes.
//
// go-ml does not ship a DummyClassifier; the ensemble package registers its
// forests exactly this way.

import (
	"encoding/json"
	"fmt"
	"log"

	goml "github.com/heru-opensource/go-ml"
)

// dummyClassifier predicts fixed class priors, whatever the input.
type dummyClassifier struct {
	nFeatures    int
	classes      []float64
	featureNames []string
	prior        []float64
}

func (d *dummyClassifier) Type() string       { return "DummyClassifier" }
func (d *dummyClassifier) NFeatures() int     { return d.nFeatures }
func (d *dummyClassifier) Classes() []float64 { return append([]float64(nil), d.classes...) }

// FeatureNames completes goml.Model. Returning nil is allowed and means the
// export carried no names; passing them through, as here, is what lets callers
// build inputs with a goml.Assembler instead of by position.
func (d *dummyClassifier) FeatureNames() []string {
	return append([]string(nil), d.featureNames...)
}

// PredictProba returns the prior for every sample. Validating the feature count
// (and nothing else) is the convention every model here follows.
func (d *dummyClassifier) PredictProba(X [][]float64) ([][]float64, error) {
	out := make([][]float64, len(X))
	for i, x := range X {
		if len(x) != d.nFeatures {
			return nil, fmt.Errorf("%w: sample %d has %d, want %d",
				goml.ErrNumFeatures, i, len(x), d.nFeatures)
		}
		out[i] = append([]float64(nil), d.prior...)
	}
	return out, nil
}

func (d *dummyClassifier) Predict(X [][]float64) ([]float64, error) {
	proba, err := d.PredictProba(X)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(proba))
	for i, p := range proba {
		best := 0
		for c := range p {
			if p[c] > p[best] {
				best = c
			}
		}
		out[i] = d.classes[best]
	}
	return out, nil
}

// decodeDummy builds the model from the type-specific "model" object of an
// export envelope. The exporter on the Python side (see tools/sklexport) writes
// the matching JSON.
func decodeDummy(raw json.RawMessage) (goml.Model, error) {
	var j struct {
		NFeatures    int       `json:"n_features"`
		Classes      []float64 `json:"classes"`
		FeatureNames []string  `json:"feature_names"`
		Prior        []float64 `json:"prior"`
	}
	if err := json.Unmarshal(raw, &j); err != nil {
		return nil, err
	}
	if len(j.Classes) != len(j.Prior) {
		return nil, fmt.Errorf("dummy: %d classes but %d priors", len(j.Classes), len(j.Prior))
	}
	if n := len(j.FeatureNames); n > 0 && n != j.NFeatures {
		return nil, fmt.Errorf("dummy: %d feature names for %d features", n, j.NFeatures)
	}
	return &dummyClassifier{
		nFeatures:    j.NFeatures,
		classes:      j.Classes,
		featureNames: j.FeatureNames,
		prior:        j.Prior,
	}, nil
}

// Registration is global and permanent, so it belongs in an init function —
// which is why importing a model package for its side effect is all a caller
// has to do.
func init() {
	goml.Register("DummyClassifier", decodeDummy)
}

// Example_customModel loads an export of the type registered above. Nothing in
// the loading code knows about dummyClassifier: goml.Load dispatches on the
// envelope's "type", exactly as image.Decode dispatches on a magic number.
func Example_customModel() {
	const export = `{
	  "format": "go-ml/v1",
	  "type": "DummyClassifier",
	  "model": {
	    "n_features": 4, "classes": [0, 1, 2], "prior": [0.6, 0.3, 0.1],
	    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
	  }
	}`

	clf, err := goml.LoadClassifierBytes([]byte(export))
	if err != nil {
		log.Fatal(err)
	}

	proba, _ := clf.PredictProba([][]float64{{5.1, 3.5, 1.4, 0.2}})
	labels, _ := clf.Predict([][]float64{{5.1, 3.5, 1.4, 0.2}})

	fmt.Println(clf.Type(), clf.NFeatures(), clf.Classes())
	fmt.Println(clf.FeatureNames())
	fmt.Printf("proba = %.1f  label = %v\n", proba[0], labels[0])
	// Output:
	// DummyClassifier 4 [0 1 2]
	// [sepal_length sepal_width petal_length petal_width]
	// proba = [0.6 0.3 0.1]  label = 0
}
