package goml_test

import (
	"math"
	"path/filepath"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	bundlemodels "github.com/heru-opensource/go-ml/examples/bundle/models"
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

// TestStaticBundleMatchesExport validates the same path for a multi-model
// bundle: every model and every metadata value compiled into the binary must be
// exactly what the JSON document holds. The members are validated against
// scikit-learn by the tests above, since they are ordinary exports; what this
// pins is that bundling and code generation preserve them, bit for bit, along
// with the tuned numbers that are the point of shipping them together.
func TestStaticBundleMatchesExport(t *testing.T) {
	loaded, err := goml.LoadBundleFile(filepath.Join("testdata", "models", "iris_bundle.json"))
	if err != nil {
		t.Fatal(err)
	}
	static := bundlemodels.IrisCascade

	if !equalStrings(static.Names(), loaded.Names()) {
		t.Fatalf("models = %v, want %v", static.Names(), loaded.Names())
	}
	if !equalStrings(static.MetaKeys(), loaded.MetaKeys()) {
		t.Fatalf("metadata keys = %v, want %v", static.MetaKeys(), loaded.MetaKeys())
	}

	// Iris inputs, including the missing values and float32-boundary rows.
	var fx fixture
	readJSON(t, filepath.Join("testdata", "fixtures", "iris.fixture.json"), &fx)
	X := rows(fx.X)

	for _, name := range loaded.Names() {
		want, err := loaded.Classifier(name)
		if err != nil {
			t.Fatal(err)
		}
		got, err := static.Classifier(name)
		if err != nil {
			t.Fatal(err)
		}
		if got.Type() != want.Type() {
			t.Errorf("%s: type %s, want %s", name, got.Type(), want.Type())
		}
		if !equalStrings(got.FeatureNames(), want.FeatureNames()) {
			t.Errorf("%s: feature names %v, want %v", name, got.FeatureNames(), want.FeatureNames())
		}

		wantProba, err := want.PredictProba(X)
		if err != nil {
			t.Fatal(err)
		}
		gotProba, err := got.PredictProba(X)
		if err != nil {
			t.Fatal(err)
		}
		for i := range wantProba {
			// Bit-for-bit: same trees, same summation order, so anything else
			// means the generator lost a value.
			if !floatsEqual(gotProba[i], wantProba[i]) {
				t.Fatalf("%s: row %d = %v, want %v", name, i, gotProba[i], wantProba[i])
			}
		}
	}

	for _, key := range loaded.MetaKeys() {
		var want, got any
		if err := loaded.Meta(key, &want); err != nil {
			t.Fatal(err)
		}
		if err := static.Meta(key, &got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("metadata %q = %v, want %v", key, got, want)
		}
	}
	t.Logf("static bundle: %d models and %d metadata keys identical to the export",
		len(loaded.Names()), len(loaded.MetaKeys()))
}

func equalStrings(a, b []string) bool {
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
