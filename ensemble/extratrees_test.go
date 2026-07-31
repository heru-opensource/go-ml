package ensemble

import (
	"errors"
	"math"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/tree"
)

// tinyExtraTrees is the ExtraTreesClassifier counterpart of tinyForest, built
// from the same stumps (see randomforest_test.go).
func tinyExtraTrees(t *testing.T, classes []float64) *ExtraTreesClassifier {
	t.Helper()
	et, err := NewExtraTreesClassifier(2, classes, []*tree.Tree{oneSplit(0), oneSplit(1)})
	if err != nil {
		t.Fatal(err)
	}
	return et
}

func TestExtraTreesPredictProbaAndPredict(t *testing.T) {
	et := tinyExtraTrees(t, []float64{0, 1})
	X := [][]float64{{0, 0}, {1, 1}, {1, 0}}
	proba, err := et.PredictProba(X)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{{1, 0}, {0, 1}, {0.5, 0.5}}
	for i := range want {
		for c := range want[i] {
			if math.Abs(proba[i][c]-want[i][c]) > 1e-15 {
				t.Errorf("proba[%d] = %v, want %v", i, proba[i], want[i])
			}
		}
	}
	pred, _ := et.Predict(X)
	// {0,0}->0, {1,1}->1, {1,0}-> tie resolved to lowest column -> class 0.
	for i, w := range []float64{0, 1, 0} {
		if pred[i] != w {
			t.Errorf("pred[%d] = %v, want %v", i, pred[i], w)
		}
	}
}

// TestExtraTreesMatchesForestArithmetic pins down the reason the two estimators
// share an implementation: given the same fitted trees they must predict
// identically, because scikit-learn's prediction path does not distinguish
// them. Only the way the trees were grown differs, and that is already fixed by
// the time a model is exported.
func TestExtraTreesMatchesForestArithmetic(t *testing.T) {
	classes := []float64{0, 1}
	trees := []*tree.Tree{oneSplit(0), oneSplit(1)}
	rf, err := NewRandomForestClassifier(2, classes, trees)
	if err != nil {
		t.Fatal(err)
	}
	et, err := NewExtraTreesClassifier(2, classes, trees)
	if err != nil {
		t.Fatal(err)
	}
	X := [][]float64{{0, 0}, {1, 1}, {1, 0}, {math.NaN(), 0.7}}
	rfProba, _ := rf.PredictProba(X)
	etProba, _ := et.PredictProba(X)
	for i := range rfProba {
		for c := range rfProba[i] {
			if rfProba[i][c] != etProba[i][c] {
				t.Errorf("row %d: extra trees %v != random forest %v", i, etProba[i], rfProba[i])
			}
		}
	}
}

func TestExtraTreesSingleRowHelpers(t *testing.T) {
	et := tinyExtraTrees(t, []float64{10, 20}) // arbitrary labels
	p, err := et.PredictProbaRow([]float64{1, 1})
	if err != nil || p[1] != 1 {
		t.Errorf("PredictProbaRow = %v, %v", p, err)
	}
	c, err := et.PredictRow([]float64{0, 0})
	if err != nil || c != 10 {
		t.Errorf("PredictRow = %v, %v", c, err)
	}
}

func TestExtraTreesWrongFeatureCount(t *testing.T) {
	et := tinyExtraTrees(t, []float64{0, 1})
	_, err := et.PredictProba([][]float64{{1, 2, 3}})
	if !errors.Is(err, goml.ErrNumFeatures) {
		t.Errorf("err = %v, want ErrNumFeatures", err)
	}
}

func TestExtraTreesConstructorValidation(t *testing.T) {
	good := oneSplit(0)
	cases := map[string]struct {
		nf      int
		classes []float64
		trees   []*tree.Tree
	}{
		"zero features":  {0, []float64{0, 1}, []*tree.Tree{good}},
		"one class":      {2, []float64{0}, []*tree.Tree{good}},
		"no trees":       {2, []float64{0, 1}, nil},
		"nil tree":       {2, []float64{0, 1}, []*tree.Tree{nil}},
		"width mismatch": {2, []float64{0, 1, 2}, []*tree.Tree{good}}, // good has width 2
	}
	for name, c := range cases {
		if _, err := NewExtraTreesClassifier(c.nf, c.classes, c.trees); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func TestExtraTreesMetadata(t *testing.T) {
	et := tinyExtraTrees(t, []float64{0, 1})
	if et.Type() != TypeExtraTreesClassifier {
		t.Errorf("Type = %q", et.Type())
	}
	if et.NFeatures() != 2 || et.NTrees() != 2 {
		t.Errorf("NFeatures=%d NTrees=%d", et.NFeatures(), et.NTrees())
	}
	// Classes returns a copy: mutating it must not affect the model.
	cs := et.Classes()
	cs[0] = 999
	if et.Classes()[0] == 999 {
		t.Error("Classes did not return a defensive copy")
	}
}
