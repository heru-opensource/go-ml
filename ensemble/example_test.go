package ensemble_test

import (
	"fmt"
	"log"
	"math"
	"testing"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/ensemble"
)

// A minimal go-ml/v1 export: a one-tree forest over a single feature that
// predicts class 0 when feature[0] <= 0.5 and class 1 otherwise. Real exports
// are produced from scikit-learn by tools/sklexport.
const miniForest = `{
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

// Example loads a model from a go-ml export and predicts, exactly mirroring
// scikit-learn's predict_proba / predict API.
func Example() {
	clf, err := goml.LoadClassifierBytes([]byte(miniForest))
	if err != nil {
		log.Fatal(err)
	}

	proba, _ := clf.PredictProba([][]float64{{0.2}, {0.9}})
	fmt.Printf("proba[0] = %.1f\n", proba[0])
	fmt.Printf("proba[1] = %.1f\n", proba[1])

	pred, _ := clf.Predict([][]float64{{0.2}, {0.9}})
	fmt.Printf("predict  = %v\n", pred)
	// Output:
	// proba[0] = [1.0 0.0]
	// proba[1] = [0.0 1.0]
	// predict  = [0 1]
}

// ExampleRandomForestClassifier_missing shows that a missing feature is encoded
// as math.NaN and routed natively by the trees (here, to the right child).
func ExampleRandomForestClassifier_missing() {
	rf, _ := ensemble.LoadRandomForestClassifier([]byte(miniForest))
	p, _ := rf.PredictProbaRow([]float64{math.NaN()})
	fmt.Printf("%.1f\n", p)
	// Output:
	// [0.0 1.0]
}

func TestLoadEnvelopeRoundTrip(t *testing.T) {
	rf, err := ensemble.LoadRandomForestClassifier([]byte(miniForest), ensemble.WithWorkers(2))
	if err != nil {
		t.Fatal(err)
	}
	if rf.Type() != ensemble.TypeRandomForestClassifier || rf.NTrees() != 1 {
		t.Fatalf("unexpected model: type=%s trees=%d", rf.Type(), rf.NTrees())
	}
	p, _ := rf.PredictProbaRow([]float64{0.9})
	if p[1] != 1 {
		t.Errorf("proba = %v, want [0 1]", p)
	}
}
