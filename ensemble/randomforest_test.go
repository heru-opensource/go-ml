package ensemble

import (
	"errors"
	"math"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/tree"
)

// oneSplit returns a stump that emits class 0 when feature f <= 0.5, else
// class 1 (each leaf is a one-hot class distribution).
func oneSplit(f int32) *tree.Tree {
	return &tree.Tree{
		Left: []int32{1, -1, -1}, Right: []int32{2, -1, -1},
		Feature: []int32{f, -2, -2}, Threshold: []float64{0.5, -2, -2},
		MissingLeft: []bool{false, false, false},
		Value:       []float64{0.5, 0.5, 1, 0, 0, 1}, ValueWidth: 2,
	}
}

func tinyForest(t *testing.T, classes []float64) *RandomForestClassifier {
	t.Helper()
	rf, err := NewRandomForestClassifier(2, classes, []*tree.Tree{oneSplit(0), oneSplit(1)})
	if err != nil {
		t.Fatal(err)
	}
	return rf
}

func TestPredictProbaAndPredict(t *testing.T) {
	rf := tinyForest(t, []float64{0, 1})
	X := [][]float64{{0, 0}, {1, 1}, {1, 0}}
	proba, err := rf.PredictProba(X)
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
	pred, _ := rf.Predict(X)
	// {0,0}->0, {1,1}->1, {1,0}-> tie resolved to lowest column -> class 0.
	for i, w := range []float64{0, 1, 0} {
		if pred[i] != w {
			t.Errorf("pred[%d] = %v, want %v", i, pred[i], w)
		}
	}
}

func TestPredictMapsClassLabels(t *testing.T) {
	rf := tinyForest(t, []float64{10, 20}) // arbitrary labels
	pred, _ := rf.Predict([][]float64{{0, 0}, {1, 1}})
	if pred[0] != 10 || pred[1] != 20 {
		t.Errorf("labels = %v, want [10 20]", pred)
	}
}

func TestSingleRowHelpers(t *testing.T) {
	rf := tinyForest(t, []float64{0, 1})
	p, err := rf.PredictProbaRow([]float64{1, 1})
	if err != nil || p[1] != 1 {
		t.Errorf("PredictProbaRow = %v, %v", p, err)
	}
	c, err := rf.PredictRow([]float64{0, 0})
	if err != nil || c != 0 {
		t.Errorf("PredictRow = %v, %v", c, err)
	}
}

func TestWrongFeatureCount(t *testing.T) {
	rf := tinyForest(t, []float64{0, 1})
	_, err := rf.PredictProba([][]float64{{1, 2, 3}})
	if !errors.Is(err, goml.ErrNumFeatures) {
		t.Errorf("err = %v, want ErrNumFeatures", err)
	}
}

func TestNaNRoutedNatively(t *testing.T) {
	// missingLeft defaults to false on oneSplit, so a NaN routes right (class 1).
	rf := tinyForest(t, []float64{0, 1})
	p, err := rf.PredictProbaRow([]float64{math.NaN(), math.NaN()})
	if err != nil {
		t.Fatal(err)
	}
	if p[1] != 1 {
		t.Errorf("NaN routing proba = %v, want [0 1]", p)
	}
}

func TestConstructorValidation(t *testing.T) {
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
		if _, err := NewRandomForestClassifier(c.nf, c.classes, c.trees); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func TestWorkerStrategiesEquivalentSmall(t *testing.T) {
	// Build a wide forest and many rows so the parallel paths actually engage,
	// then confirm they match the forced-sequential result.
	trees := make([]*tree.Tree, 64)
	for i := range trees {
		trees[i] = oneSplit(int32(i % 2))
	}
	mk := func(w int) *RandomForestClassifier {
		rf, err := NewRandomForestClassifier(2, []float64{0, 1}, trees, WithWorkers(w))
		if err != nil {
			t.Fatal(err)
		}
		return rf
	}
	X := make([][]float64, 2000)
	for i := range X {
		X[i] = []float64{float64(i % 3), float64((i + 1) % 3)}
	}
	seq, _ := mk(1).PredictProba(X)
	for _, w := range []int{2, 4, 8} {
		got, _ := mk(w).PredictProba(X)
		for i := range got {
			for c := range got[i] {
				if math.Abs(got[i][c]-seq[i][c]) > 1e-15 {
					t.Fatalf("workers=%d row %d differs: %v vs %v", w, i, got[i], seq[i])
				}
			}
		}
	}
}

func TestMetadata(t *testing.T) {
	rf := tinyForest(t, []float64{0, 1})
	if rf.Type() != TypeRandomForestClassifier {
		t.Errorf("Type = %q", rf.Type())
	}
	if rf.NFeatures() != 2 || rf.NTrees() != 2 {
		t.Errorf("NFeatures=%d NTrees=%d", rf.NFeatures(), rf.NTrees())
	}
	// Classes returns a copy: mutating it must not affect the model.
	cs := rf.Classes()
	cs[0] = 999
	if rf.Classes()[0] == 999 {
		t.Error("Classes did not return a defensive copy")
	}
}
