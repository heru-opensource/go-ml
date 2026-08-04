package ensemble

import (
	"strings"
	"testing"

	"github.com/heru-opensource/go-ml/tree"
)

// namedForest is miniForest (see example_test.go) with feature names attached.
const namedForest = `{
  "format": "go-ml/v1",
  "type": "RandomForestClassifier",
  "model": {
    "n_features": 1, "n_outputs": 1, "classes": [0, 1],
    "feature_names": ["petal_length"],
    "trees": [{
      "node_count": 3, "value_width": 2,
      "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
      "threshold": [0.5, -2, -2], "missing_left": [false, false, false],
      "value": [0.5, 0.5, 1, 0, 0, 1]
    }]
  }
}`

func TestFeatureNamesFromExport(t *testing.T) {
	rf, err := LoadRandomForestClassifier([]byte(namedForest))
	if err != nil {
		t.Fatal(err)
	}
	names := rf.FeatureNames()
	if len(names) != 1 || names[0] != "petal_length" {
		t.Fatalf("FeatureNames = %v, want [petal_length]", names)
	}
	// A copy, like Classes: mutating it must not reach the model.
	names[0] = "clobbered"
	if rf.FeatureNames()[0] != "petal_length" {
		t.Error("FeatureNames did not return a defensive copy")
	}
}

// TestFeatureNamesAbsentIsFine pins the compatibility promise: an export from
// an estimator fitted on a bare array carries no names, and nothing breaks.
func TestFeatureNamesAbsentIsFine(t *testing.T) {
	rf := tinyForest(t, []float64{0, 1})
	if got := rf.FeatureNames(); len(got) != 0 {
		t.Errorf("FeatureNames = %v, want empty", got)
	}
}

func TestWithFeatureNames(t *testing.T) {
	trees := []*tree.Tree{oneSplit(0), oneSplit(1)}

	rf, err := NewRandomForestClassifier(2, []float64{0, 1}, trees,
		WithFeatureNames([]string{"left", "right"}))
	if err != nil {
		t.Fatal(err)
	}
	if got := rf.FeatureNames(); len(got) != 2 || got[1] != "right" {
		t.Errorf("FeatureNames = %v", got)
	}

	// The count must match, or a caller could address the wrong column by name.
	_, err = NewRandomForestClassifier(2, []float64{0, 1}, trees,
		WithFeatureNames([]string{"only_one"}))
	if err == nil || !strings.Contains(err.Error(), "feature names") {
		t.Errorf("err = %v, want a feature-name count error", err)
	}
}

func TestLoadForestFeatureNamesOverride(t *testing.T) {
	f, err := LoadForest([]byte(namedForest), WithFeatureNames([]string{"renamed"}))
	if err != nil {
		t.Fatal(err)
	}
	if got := f.FeatureNames(); len(got) != 1 || got[0] != "renamed" {
		t.Errorf("FeatureNames = %v, want [renamed]", got)
	}

	if _, err := LoadForest([]byte(namedForest), WithFeatureNames([]string{"a", "b"})); err == nil {
		t.Error("a wrong-length override should be rejected")
	}
}
