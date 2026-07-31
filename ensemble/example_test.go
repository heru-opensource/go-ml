package ensemble_test

import (
	"fmt"
	"log"
	"math"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/ensemble"
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

// The same export as an ExtraTreesClassifier: the "type" field is what selects
// the estimator, and the "model" payload is identical for both — one tree here
// with an uneven leaf distribution, as class_weight="balanced" would produce.
const miniExtraTrees = `{
  "format": "go-ml/v1",
  "type": "ExtraTreesClassifier",
  "model": {
    "n_features": 1, "n_outputs": 1, "classes": [0, 1],
    "trees": [{
      "node_count": 3, "value_width": 2,
      "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
      "threshold": [0.5, -2, -2], "missing_left": [true, false, false],
      "value": [0.5, 0.5, 0.8, 0.2, 0.25, 0.75]
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

// ExampleExtraTreesClassifier loads an extremely randomized trees model. It is
// the same code as for a random forest — only the export's "type" differs —
// which is the point of the goml.Classifier interface.
func ExampleExtraTreesClassifier() {
	et, err := ensemble.LoadExtraTreesClassifier([]byte(miniExtraTrees))
	if err != nil {
		log.Fatal(err)
	}
	p, _ := et.PredictProbaRow([]float64{0.9})
	fmt.Printf("%s: %.2f -> %.2f\n", et.Type(), 0.9, p)
	// Output:
	// ExtraTreesClassifier: 0.90 -> [0.25 0.75]
}

// ExampleLoadForest shows the type-agnostic loader: it accepts either estimator
// this package implements, so code that only needs to predict does not have to
// know which one it was handed.
func ExampleLoadForest() {
	for _, export := range [][]byte{[]byte(miniForest), []byte(miniExtraTrees)} {
		f, err := ensemble.LoadForest(export, ensemble.WithWorkers(1))
		if err != nil {
			log.Fatal(err)
		}
		c, _ := f.PredictRow([]float64{0.9})
		fmt.Printf("%-22s trees=%d class=%g\n", f.Type(), f.NTrees(), c)
	}
	// Output:
	// RandomForestClassifier trees=1 class=1
	// ExtraTreesClassifier   trees=1 class=1
}

// ExampleWithWorkers caps the goroutines a single prediction call may use.
// Large batches and large forests are split across goroutines by default
// (GOMAXPROCS); WithWorkers(1) forces the sequential path, which is the one
// that is bit-for-bit identical to scikit-learn's single-threaded summation.
// Every setting agrees to floating-point rounding, so this is a throughput
// knob, not a correctness one.
func ExampleWithWorkers() {
	seq, err := ensemble.LoadForest([]byte(miniForest), ensemble.WithWorkers(1))
	if err != nil {
		log.Fatal(err)
	}
	par, err := ensemble.LoadForest([]byte(miniForest), ensemble.WithWorkers(4))
	if err != nil {
		log.Fatal(err)
	}

	a, _ := seq.PredictProbaRow([]float64{0.9})
	b, _ := par.PredictProbaRow([]float64{0.9})
	fmt.Printf("%.1f == %.1f: %v\n", a, b, a[0] == b[0] && a[1] == b[1])
	// Output:
	// [0.0 1.0] == [0.0 1.0]: true
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

	et, err := ensemble.LoadExtraTreesClassifier([]byte(miniExtraTrees), ensemble.WithWorkers(2))
	if err != nil {
		t.Fatal(err)
	}
	if et.Type() != ensemble.TypeExtraTreesClassifier || et.NTrees() != 1 {
		t.Fatalf("unexpected model: type=%s trees=%d", et.Type(), et.NTrees())
	}
	// missing_left is set on the root here, so a NaN takes the left branch.
	p, _ = et.PredictProbaRow([]float64{math.NaN()})
	if p[0] != 0.8 {
		t.Errorf("NaN routing proba = %v, want [0.8 0.2]", p)
	}
}

// TestConcreteLoadersRejectOtherTypes guards against silently treating one
// estimator's export as the other's: the concrete loaders are typed, and the
// generic goml.Load dispatches on the envelope's "type".
func TestConcreteLoadersRejectOtherTypes(t *testing.T) {
	if _, err := ensemble.LoadExtraTreesClassifier([]byte(miniForest)); err == nil {
		t.Error("LoadExtraTreesClassifier accepted a RandomForestClassifier export")
	}
	if _, err := ensemble.LoadRandomForestClassifier([]byte(miniExtraTrees)); err == nil {
		t.Error("LoadRandomForestClassifier accepted an ExtraTreesClassifier export")
	}

	m, err := goml.LoadClassifierBytes([]byte(miniExtraTrees))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.(*ensemble.ExtraTreesClassifier); !ok {
		t.Errorf("goml.Load produced %T (%s), want *ensemble.ExtraTreesClassifier", m, m.Type())
	}
}
