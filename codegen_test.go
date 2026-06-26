package goml_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/heru-public/go-ml/examples/classify/models"
)

// TestStaticModelMatchesSklearn validates the static code-generation path: the
// model compiled into the binary by go-ml-gen (see examples/classify/models)
// must produce exactly what scikit-learn produced, just like the JSON-loaded
// model. This proves the generator preserves every node and float64 value.
func TestStaticModelMatchesSklearn(t *testing.T) {
	clf := models.Iris

	var fx fixture
	readJSON(t, filepath.Join("testdata", "fixtures", "iris.fixture.json"), &fx)

	if clf.NFeatures() != fx.NFeatures {
		t.Fatalf("NFeatures = %d, want %d", clf.NFeatures(), fx.NFeatures)
	}
	proba, err := clf.PredictProba(rows(fx.X))
	if err != nil {
		t.Fatalf("PredictProba: %v", err)
	}
	var maxDiff float64
	for i := range proba {
		for c := range proba[i] {
			if d := math.Abs(proba[i][c] - fx.PredictProba[i][c]); d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxDiff > 1e-9 {
		t.Errorf("static predict_proba max abs diff %.3g exceeds tol", maxDiff)
	}
	pred, _ := clf.Predict(rows(fx.X))
	for i := range pred {
		if pred[i] != fx.Predict[i] {
			t.Errorf("static predict[%d] = %v, want %v", i, pred[i], fx.Predict[i])
		}
	}
	t.Logf("static iris: predict_proba max abs diff = %.3g", maxDiff)
}
