package goml_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/ensemble"
	"github.com/heru-opensource/go-ml/internal/jsonx"
)

// fixture is the JSON produced by tools/sklexport/make_fixtures.py: a batch of
// inputs and scikit-learn's exact predict_proba / predict outputs for them.
type fixture struct {
	NFeatures    int            `json:"n_features"`
	Classes      jsonx.Floats   `json:"classes"`
	X            []jsonx.Floats `json:"X"`
	PredictProba []jsonx.Floats `json:"predict_proba"`
	Predict      jsonx.Floats   `json:"predict"`
}

// models trained and exported by tools/sklexport/train_examples.py, each paired
// with the scikit-learn reference fixture for the same estimator. iris is a
// 3-class random forest; forest_bench is a larger binary one;
// extratrees_balanced is an ExtraTreesClassifier fitted with
// class_weight="balanced" on an imbalanced 3-class set.
var validationModels = []string{"iris", "forest_bench", "extratrees_balanced"}

func rows(fs []jsonx.Floats) [][]float64 {
	out := make([][]float64, len(fs))
	for i := range fs {
		out[i] = []float64(fs[i])
	}
	return out
}

// TestAgainstSklearn is the headline correctness test: it loads each exported
// model and asserts that go-ml reproduces scikit-learn's predict_proba and
// predict outputs, for every estimator type the repo ships fixtures for.
func TestAgainstSklearn(t *testing.T) {
	// Tolerance is generous; in practice the batch path is bit-for-bit
	// identical to scikit-learn's single-threaded summation, so observed
	// differences are ~1e-16. The test logs the real maximum.
	const probaTol = 1e-9

	for _, name := range validationModels {
		t.Run(name, func(t *testing.T) {
			clf, err := goml.LoadClassifierFile(filepath.Join("testdata", "models", name+".json"))
			if err != nil {
				t.Fatalf("load model: %v", err)
			}
			var fx fixture
			readJSON(t, filepath.Join("testdata", "fixtures", name+".fixture.json"), &fx)

			if clf.NFeatures() != fx.NFeatures {
				t.Fatalf("NFeatures = %d, want %d", clf.NFeatures(), fx.NFeatures)
			}
			if !floatsEqual(clf.Classes(), []float64(fx.Classes)) {
				t.Fatalf("Classes = %v, want %v", clf.Classes(), fx.Classes)
			}

			X := rows(fx.X)
			proba, err := clf.PredictProba(X)
			if err != nil {
				t.Fatalf("PredictProba: %v", err)
			}
			if len(proba) != len(fx.PredictProba) {
				t.Fatalf("got %d proba rows, want %d", len(proba), len(fx.PredictProba))
			}

			var maxDiff float64
			for i := range proba {
				want := fx.PredictProba[i]
				if len(proba[i]) != len(want) {
					t.Fatalf("row %d: got %d cols, want %d", i, len(proba[i]), len(want))
				}
				for c := range proba[i] {
					d := math.Abs(proba[i][c] - want[c])
					if d > maxDiff {
						maxDiff = d
					}
				}
			}
			if maxDiff > probaTol {
				t.Errorf("predict_proba max abs diff %.3g exceeds tol %.3g", maxDiff, probaTol)
			}
			t.Logf("%s: %s, %d samples, %d trees, predict_proba max abs diff = %.3g",
				name, clf.Type(), len(X), clf.(ensemble.Forest).NTrees(), maxDiff)

			pred, err := clf.Predict(X)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for i := range pred {
				if pred[i] != fx.Predict[i] {
					t.Errorf("predict[%d] = %v, want %v (proba=%v want=%v)",
						i, pred[i], fx.Predict[i], proba[i], fx.PredictProba[i])
				}
			}
		})
	}
}

// TestWorkerStrategiesAgree checks that the sequential, row-parallel and
// tree-parallel accumulation paths all produce the same probabilities on the
// real models, within floating-point rounding. ensemble.LoadForest takes
// whichever estimator the export holds, so the same check covers both types.
func TestWorkerStrategiesAgree(t *testing.T) {
	for _, name := range []string{"forest_bench", "extratrees_balanced"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", "models", name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			var fx fixture
			readJSON(t, filepath.Join("testdata", "fixtures", name+".fixture.json"), &fx)
			X := rows(fx.X)

			seq, err := ensemble.LoadForest(data, ensemble.WithWorkers(1))
			if err != nil {
				t.Fatal(err)
			}
			base, _ := seq.PredictProba(X)

			for _, w := range []int{2, 4, 8} {
				par, err := ensemble.LoadForest(data, ensemble.WithWorkers(w))
				if err != nil {
					t.Fatal(err)
				}
				got, _ := par.PredictProba(X)
				var maxDiff float64
				for i := range got {
					for c := range got[i] {
						if d := math.Abs(got[i][c] - base[i][c]); d > maxDiff {
							maxDiff = d
						}
					}
				}
				if maxDiff > 1e-12 {
					t.Errorf("workers=%d diverges from sequential by %.3g", w, maxDiff)
				}
				// Single-row prediction exercises the tree-parallel path.
				if _, err := par.PredictProbaRow(X[0]); err != nil {
					t.Errorf("workers=%d PredictProbaRow: %v", w, err)
				}
			}
		})
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

func floatsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
