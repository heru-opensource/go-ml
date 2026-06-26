// Command classify is a runnable example of using a go-ml model that has been
// compiled statically into the binary with go-ml-gen — there is no Python, no
// pickle, and no model file at runtime.
//
// The model here is a RandomForestClassifier trained on the Iris dataset (4
// features, 3 classes); see examples/classify/models for the generated source.
// The same code works for any classifier: program against the goml.Classifier
// interface and a missing feature is just math.NaN.
//
// Run it with:
//
//	go run ./examples/classify
package main

import (
	"fmt"
	"math"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/examples/classify/models"
)

func main() {
	// models.Iris is a *ensemble.RandomForestClassifier, but we use it through
	// the generic goml.Classifier interface to show the model-agnostic API.
	var clf goml.Classifier = models.Iris

	fmt.Printf("%s — %d features, classes %v\n\n", clf.Type(), clf.NFeatures(), clf.Classes())

	samples := [][]float64{
		{5.1, 3.5, 1.4, 0.2},        // typical setosa
		{6.7, 3.0, 5.2, 2.3},        // typical virginica
		{5.9, 3.0, math.NaN(), 1.8}, // a missing feature (handled natively)
	}

	proba, err := clf.PredictProba(samples)
	if err != nil {
		panic(err)
	}
	labels, _ := clf.Predict(samples)

	for i, x := range samples {
		fmt.Printf("features=%v\n  class=%g  proba=%s\n", x, labels[i], formatProba(clf.Classes(), proba[i]))
	}
}

func formatProba(classes, p []float64) string {
	s := "["
	for i := range p {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%g:%.3f", classes[i], p[i])
	}
	return s + "]"
}
