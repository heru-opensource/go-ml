// Command classify is a runnable example of using go-ml models that have been
// compiled statically into the binary with go-ml-gen — there is no Python, no
// pickle, and no model file at runtime.
//
// It classifies with two estimator types through one interface: a
// RandomForestClassifier trained on the Iris dataset (4 features, 3 classes)
// and an ExtraTreesClassifier fitted with class_weight="balanced" on an
// imbalanced 3-class set (10 features); see examples/classify/models for the
// generated source. The predicting code is the same for both: program against
// the goml.Classifier interface, and a missing feature is just math.NaN.
//
// Run it with:
//
//	go run ./examples/classify
package main

import (
	"fmt"
	"math"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/examples/classify/models"
)

func main() {
	// models.Iris is a *ensemble.RandomForestClassifier and
	// models.ExtraTreesBalanced an *ensemble.ExtraTreesClassifier; both are used
	// here as a goml.Classifier, so nothing below depends on the estimator type.
	classify(models.Iris, [][]float64{
		{5.1, 3.5, 1.4, 0.2},        // typical setosa
		{6.7, 3.0, 5.2, 2.3},        // typical virginica
		{5.9, 3.0, math.NaN(), 1.8}, // a missing feature (handled natively)
	})

	// The balanced extra-trees model. Class weighting happened during training
	// in scikit-learn and is baked into the exported leaves, which is why the
	// rare third class is still predicted confidently here.
	classify(models.ExtraTreesBalanced, [][]float64{
		{0.79, -2.55, 2.26, 0.23, 0.54, 0.15, 0.08, -0.68, 0.28, -2.4},
		{-1.05, 2.53, -0.31, 2.38, 0.12, 1.68, -0.57, 1.28, -0.89, -0.82},
		{math.NaN(), 2.53, -0.31, 2.38, math.NaN(), 1.68, -0.57, 1.28, -0.89, -0.82},
	})
}

func classify(clf goml.Classifier, samples [][]float64) {
	fmt.Printf("%s — %d features, classes %v\n\n", clf.Type(), clf.NFeatures(), clf.Classes())

	proba, err := clf.PredictProba(samples)
	if err != nil {
		panic(err)
	}
	labels, _ := clf.Predict(samples)

	for i, x := range samples {
		fmt.Printf("features=%v\n  class=%g  proba=%s\n", x, labels[i], formatProba(clf.Classes(), proba[i]))
	}
	fmt.Println()
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
