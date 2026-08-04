package goml_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	goml "github.com/heru-opensource/go-ml"
)

// namedModel is the smallest thing that satisfies goml.Model, so these tests
// exercise the Assembler rather than any particular estimator.
type namedModel struct{ names []string }

func (m namedModel) Type() string           { return "NamedModel" }
func (m namedModel) NFeatures() int         { return len(m.names) }
func (m namedModel) FeatureNames() []string { return m.names }

func TestAssemblerOrdersByName(t *testing.T) {
	a, err := goml.NewAssembler(namedModel{[]string{"a", "b", "c"}})
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately not in model order: that is the whole point.
	row, err := a.Row(map[string]float64{"c": 3, "a": 1, "b": 2})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{1, 2, 3} {
		if row[i] != want {
			t.Errorf("row = %v, want [1 2 3]", row)
			break
		}
	}
}

func TestAssemblerOmittedFeatureIsMissing(t *testing.T) {
	a, _ := goml.NewAssembler(namedModel{[]string{"a", "b"}})
	row, err := a.Row(map[string]float64{"a": 1})
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(row[1]) {
		t.Errorf("row[1] = %v, want NaN", row[1])
	}
	if missing := a.Missing(map[string]float64{"a": 1}); len(missing) != 1 || missing[0] != "b" {
		t.Errorf("Missing = %v, want [b]", missing)
	}
}

// TestAssemblerRejectsUnknownFeature pins the reason this type exists: a typo or
// a stale caller must be loud, not silently dropped.
func TestAssemblerRejectsUnknownFeature(t *testing.T) {
	a, _ := goml.NewAssembler(namedModel{[]string{"a", "b"}})
	_, err := a.Row(map[string]float64{"a": 1, "bb": 2})
	if !errors.Is(err, goml.ErrUnknownFeature) {
		t.Fatalf("err = %v, want ErrUnknownFeature", err)
	}
	if !strings.Contains(err.Error(), "bb") {
		t.Errorf("error should name the offending feature; got %v", err)
	}
}

func TestAssemblerRowsReportsSampleIndex(t *testing.T) {
	a, _ := goml.NewAssembler(namedModel{[]string{"a"}})
	_, err := a.Rows([]map[string]float64{{"a": 1}, {"nope": 2}})
	if err == nil || !strings.Contains(err.Error(), "sample 1") {
		t.Errorf("err = %v, want it to name sample 1", err)
	}
}

func TestNewAssemblerRequiresNames(t *testing.T) {
	if _, err := goml.NewAssembler(namedModel{}); !errors.Is(err, goml.ErrNoFeatureNames) {
		t.Errorf("err = %v, want ErrNoFeatureNames", err)
	}
	if _, err := goml.NewAssembler(namedModel{[]string{"a", "a"}}); err == nil {
		t.Error("duplicate feature names should be rejected")
	}
}

// TestAssemblerMatchesPositionalOrder is the end-to-end claim: assembling by
// name from a scrambled map produces exactly what hand-ordering the vector
// correctly would, on a real exported model.
func TestAssemblerMatchesPositionalOrder(t *testing.T) {
	clf, err := goml.LoadClassifierFile("testdata/models/iris.json")
	if err != nil {
		t.Fatal(err)
	}
	names := clf.FeatureNames()
	if len(names) != clf.NFeatures() {
		t.Fatalf("iris export should carry %d feature names, got %v", clf.NFeatures(), names)
	}

	a, err := goml.NewAssembler(clf)
	if err != nil {
		t.Fatal(err)
	}
	positional := []float64{5.1, 3.5, 1.4, 0.2}
	byName := map[string]float64{}
	for i, n := range names {
		byName[n] = positional[i]
	}

	want, err := clf.PredictProba([][]float64{positional})
	if err != nil {
		t.Fatal(err)
	}
	row, err := a.Row(byName)
	if err != nil {
		t.Fatal(err)
	}
	got, err := clf.PredictProba([][]float64{row})
	if err != nil {
		t.Fatal(err)
	}
	if !floatsEqual(got[0], want[0]) {
		t.Errorf("assembled prediction %v != positional %v", got[0], want[0])
	}
}
