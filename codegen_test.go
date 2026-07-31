package goml_test

import (
	"math"
	"path/filepath"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/examples/classify/models"
)

// TestStaticModelsMatchSklearn validates the static code-generation path: the
// models compiled into the binary by go-ml-gen (see examples/classify/models)
// must produce exactly what scikit-learn produced, just like the JSON-loaded
// models. This proves the generator preserves every node and float64 value, for
// each estimator type it can emit.
func TestStaticModelsMatchSklearn(t *testing.T) {
	static := []struct {
		name string
		clf  goml.Classifier
	}{
		{"iris", models.Iris},
		{"extratrees_balanced", models.ExtraTreesBalanced},
	}
	for _, tc := range static {
		name, clf := tc.name, tc.clf
		t.Run(name, func(t *testing.T) {
			var fx fixture
			readJSON(t, filepath.Join("testdata", "fixtures", name+".fixture.json"), &fx)

			if clf.NFeatures() != fx.NFeatures {
				t.Fatalf("NFeatures = %d, want %d", clf.NFeatures(), fx.NFeatures)
			}
			if !floatsEqual(clf.Classes(), []float64(fx.Classes)) {
				t.Fatalf("Classes = %v, want %v", clf.Classes(), fx.Classes)
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
			t.Logf("static %s (%s): predict_proba max abs diff = %.3g", name, clf.Type(), maxDiff)
		})
	}
}
