package goml_test

import (
	"errors"
	"fmt"
	"log"
	"math"

	goml "github.com/heru-opensource/go-ml"
	_ "github.com/heru-opensource/go-ml/ensemble" // registers the tree-ensemble models
)

// miniExport is a one-tree RandomForestClassifier over a single feature: class
// 0 when feature 0 is <= 0.5, class 1 above it. Real exports are produced from
// a fitted scikit-learn estimator by tools/sklexport; this one is small enough
// to read inline.
const miniExport = `{
  "format": "go-ml/v1",
  "type": "RandomForestClassifier",
  "model": {
    "n_features": 1, "n_outputs": 1, "classes": [0, 1],
    "trees": [{
      "node_count": 3, "value_width": 2,
      "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
      "threshold": [0.5, -2, -2], "missing_left": [false, false, false],
      "value": [0.5, 0.5, 1, 0, 0, 1]
    }]
  }
}`

// Example loads an exported model and predicts with it. The API mirrors
// scikit-learn: PredictProba returns per-class probabilities in Classes order,
// and Predict returns the arg-max label.
func Example() {
	clf, err := goml.LoadClassifierBytes([]byte(miniExport))
	if err != nil {
		log.Fatal(err)
	}

	X := [][]float64{{0.2}, {0.9}}
	proba, err := clf.PredictProba(X)
	if err != nil {
		log.Fatal(err)
	}
	labels, _ := clf.Predict(X)

	fmt.Println(clf.Type(), clf.NFeatures(), clf.Classes())
	fmt.Printf("proba = %.1f\n", proba)
	fmt.Printf("labels = %v\n", labels)
	// Output:
	// RandomForestClassifier 1 [0 1]
	// proba = [[1.0 0.0] [0.0 1.0]]
	// labels = [0 1]
}

// ExampleLoadClassifierFile reads an export from disk — the startup-time
// alternative to embedding it. The model here is the repository's own Iris
// forest; the two samples are a typical setosa and a typical virginica.
func ExampleLoadClassifierFile() {
	clf, err := goml.LoadClassifierFile("testdata/models/iris.json")
	if err != nil {
		log.Fatal(err)
	}

	proba, _ := clf.PredictProba([][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{6.7, 3.0, 5.2, 2.3},
	})
	labels, _ := clf.Predict([][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{6.7, 3.0, 5.2, 2.3},
	})

	for i := range labels {
		fmt.Printf("class %g  proba %.3f\n", labels[i], proba[i])
	}
	// Output:
	// class 0  proba [1.000 0.000 0.000]
	// class 2  proba [0.000 0.003 0.998]
}

// ExampleNewAssembler builds inputs by name instead of by position.
//
// Feature order is part of a model, and a vector in the wrong order is made of
// individually valid numbers — nothing downstream can catch it. When the export
// carries names (scikit-learn records them for an estimator fitted on a named
// frame), assembling by name makes that mistake impossible, and a retrain that
// reorders columns changes the export rather than silently breaking callers.
func ExampleNewAssembler() {
	clf, err := goml.LoadClassifierFile("testdata/models/iris.json")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(clf.FeatureNames())

	a, err := goml.NewAssembler(clf)
	if err != nil {
		log.Fatal(err) // the export carries no names
	}

	// Order here is deliberately not the model's, and petal_width is absent —
	// an omitted feature is a missing one (NaN), which the trees route natively.
	row, err := a.Row(map[string]float64{
		"petal_length": 1.4,
		"sepal_width":  3.5,
		"sepal_length": 5.1,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(a.Missing(map[string]float64{"petal_length": 1.4, "sepal_width": 3.5, "sepal_length": 5.1}))

	label, _ := clf.Predict([][]float64{row})
	fmt.Printf("class %g\n", label[0])

	// A name the model does not have is an error, not a silently dropped value.
	_, err = a.Row(map[string]float64{"petal_len": 1.4})
	fmt.Println(errors.Is(err, goml.ErrUnknownFeature))
	// Output:
	// [sepal_length sepal_width petal_length petal_width]
	// [petal_width]
	// class 0
	// true
}

// Example_missingFeatures shows the missing-value convention: an absent feature
// is math.NaN, and the trees route it the way scikit-learn learned to during
// fitting rather than erroring or imputing. Here the root's missing direction
// is right, which is the class-1 leaf.
func Example_missingFeatures() {
	clf, err := goml.LoadClassifierBytes([]byte(miniExport))
	if err != nil {
		log.Fatal(err)
	}

	proba, err := clf.PredictProba([][]float64{{math.NaN()}})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%.1f\n", proba[0])
	// Output:
	// [0.0 1.0]
}

// Example_errors shows the two failures worth handling explicitly. Both are
// sentinel errors, so test them with errors.Is rather than by string.
func Example_errors() {
	// An estimator type nobody registered — usually a missing import of the
	// package that implements it.
	_, err := goml.LoadBytes([]byte(`{"format":"go-ml/v1","type":"SomeFutureModel","model":{}}`))
	fmt.Println("unknown type:", errors.Is(err, goml.ErrUnknownType))

	// A sample whose length does not match the model. This is the only
	// per-call validation go-ml does.
	clf, _ := goml.LoadClassifierBytes([]byte(miniExport))
	_, err = clf.PredictProba([][]float64{{1.0, 2.0}})
	fmt.Println("wrong feature count:", errors.Is(err, goml.ErrNumFeatures))
	// Output:
	// unknown type: true
	// wrong feature count: true
}
