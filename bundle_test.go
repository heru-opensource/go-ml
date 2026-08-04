package goml_test

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	_ "github.com/heru-opensource/go-ml/ensemble" // registers the tree-ensemble models
)

// miniBundle is two one-tree forests plus the numbers a caller would have had to
// hand-write beside them. The models are the same shape as miniExport.
const miniBundle = `{
  "format": "go-ml/bundle-v1",
  "models": {
    "screen": {
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
    },
    "confirm": {
      "type": "ExtraTreesClassifier",
      "model": {
        "n_features": 1, "n_outputs": 1, "classes": [0, 1],
        "trees": [{
          "node_count": 3, "value_width": 2,
          "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
          "threshold": [0.9, -2, -2], "missing_left": [false, false, false],
          "value": [0.5, 0.5, 1, 0, 0, 1]
        }]
      }
    }
  },
  "metadata": {
    "threshold": 0.83, "max_stages": 2, "tuned_for": "specificity",
    "enabled": true, "cutoffs": [0.1, 0.9], "unbounded": "Infinity"
  }
}`

func loadMiniBundle(t *testing.T) *goml.Bundle {
	t.Helper()
	b, err := goml.LoadBundleBytes([]byte(miniBundle))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestBundleModels(t *testing.T) {
	b := loadMiniBundle(t)
	if got := b.Names(); len(got) != 2 || got[0] != "confirm" || got[1] != "screen" {
		t.Fatalf("Names = %v, want [confirm screen] (sorted)", got)
	}

	// Each model decodes to its own estimator type, exactly as a single-model
	// export would.
	screen, err := b.Classifier("screen")
	if err != nil {
		t.Fatal(err)
	}
	if screen.Type() != "RandomForestClassifier" {
		t.Errorf("screen is a %s", screen.Type())
	}
	confirm, err := b.Model("confirm")
	if err != nil {
		t.Fatal(err)
	}
	if confirm.Type() != "ExtraTreesClassifier" {
		t.Errorf("confirm is a %s", confirm.Type())
	}

	proba, err := screen.PredictProba([][]float64{{0.2}, {0.9}})
	if err != nil {
		t.Fatal(err)
	}
	if proba[0][0] != 1 || proba[1][1] != 1 {
		t.Errorf("screen proba = %v", proba)
	}
}

func TestBundleUnknownModel(t *testing.T) {
	b := loadMiniBundle(t)
	_, err := b.Model("nope")
	if !errors.Is(err, goml.ErrUnknownModel) {
		t.Fatalf("err = %v, want ErrUnknownModel", err)
	}
	if !strings.Contains(err.Error(), "screen") {
		t.Errorf("error should list what the bundle does hold; got %v", err)
	}
}

func TestBundleMetadataAccessors(t *testing.T) {
	b := loadMiniBundle(t)

	if got, err := b.Float("threshold"); err != nil || got != 0.83 {
		t.Errorf("Float(threshold) = %v, %v", got, err)
	}
	if got, err := b.Int("max_stages"); err != nil || got != 2 {
		t.Errorf("Int(max_stages) = %v, %v", got, err)
	}
	if got, err := b.String("tuned_for"); err != nil || got != "specificity" {
		t.Errorf("String(tuned_for) = %q, %v", got, err)
	}
	if got, err := b.Bool("enabled"); err != nil || !got {
		t.Errorf("Bool(enabled) = %v, %v", got, err)
	}

	// Anything the typed accessors do not cover comes back through Meta.
	var cutoffs []float64
	if err := b.Meta("cutoffs", &cutoffs); err != nil || len(cutoffs) != 2 || cutoffs[1] != 0.9 {
		t.Errorf("Meta(cutoffs) = %v, %v", cutoffs, err)
	}

	// The export format's non-finite sentinels work here too.
	if got, err := b.Float("unbounded"); err != nil || !math.IsInf(got, 1) {
		t.Errorf("Float(unbounded) = %v, %v", got, err)
	}

	if got := b.MetaKeys(); len(got) != 6 || got[0] != "cutoffs" {
		t.Errorf("MetaKeys = %v", got)
	}
}

// TestBundleMissingMetadataIsLoud is the reason bundles exist: a threshold that
// is not there must stop the program, not read as zero.
func TestBundleMissingMetadataIsLoud(t *testing.T) {
	b := loadMiniBundle(t)
	for _, get := range []func() error{
		func() error { _, err := b.Float("absent"); return err },
		func() error { _, err := b.Int("absent"); return err },
		func() error { _, err := b.String("absent"); return err },
		func() error { _, err := b.Bool("absent"); return err },
		func() error { return b.Meta("absent", new(float64)) },
	} {
		if err := get(); !errors.Is(err, goml.ErrUnknownMeta) {
			t.Errorf("err = %v, want ErrUnknownMeta", err)
		}
	}
}

func TestBundleMetadataTypeMismatch(t *testing.T) {
	b := loadMiniBundle(t)
	if _, err := b.Int("threshold"); err == nil { // 0.83 is not an integer
		t.Error("Int on a fractional value should fail rather than truncate")
	}
	if _, err := b.String("threshold"); err == nil {
		t.Error("String on a number should fail")
	}
	if _, err := b.Float("tuned_for"); err == nil {
		t.Error("Float on a non-numeric string should fail")
	}
}

// TestBundleAndModelFormatsDoNotMix checks that each loader points at the other
// when handed the wrong document, since the two are easy to confuse.
func TestBundleAndModelFormatsDoNotMix(t *testing.T) {
	_, err := goml.LoadBundleBytes([]byte(miniExport))
	if !errors.Is(err, goml.ErrFormat) || !strings.Contains(err.Error(), "use Load") {
		t.Errorf("err = %v, want an ErrFormat pointing at Load", err)
	}

	_, err = goml.LoadBytes([]byte(miniBundle))
	if !errors.Is(err, goml.ErrFormat) || !strings.Contains(err.Error(), "use LoadBundle") {
		t.Errorf("err = %v, want an ErrFormat pointing at LoadBundle", err)
	}
}

func TestNewBundleValidation(t *testing.T) {
	if _, err := goml.NewBundle(nil, nil); err == nil {
		t.Error("a bundle with no models should be rejected")
	}
	if _, err := goml.NewBundle(map[string]goml.Model{"a": nil}, nil); err == nil {
		t.Error("a nil model should be rejected")
	}
	m, err := goml.LoadClassifierBytes([]byte(miniExport))
	if err != nil {
		t.Fatal(err)
	}
	_, err = goml.NewBundle(map[string]goml.Model{"a": m},
		map[string]json.RawMessage{"bad": json.RawMessage(`{oops`)})
	if err == nil {
		t.Error("invalid JSON metadata should be rejected")
	}
}

// TestBundleFromFile exercises the shipped artifact: two Iris estimators and the
// thresholds tuned with them, as scikit-learn's exporter wrote it.
func TestBundleFromFile(t *testing.T) {
	b, err := goml.LoadBundleFile("testdata/models/iris_bundle.json")
	if err != nil {
		t.Fatal(err)
	}
	if got := b.Names(); len(got) != 2 {
		t.Fatalf("Names = %v, want two models", got)
	}
	screen, err := b.Classifier("screen")
	if err != nil {
		t.Fatal(err)
	}
	// Feature names survive bundling: each member is an ordinary export.
	if got := screen.FeatureNames(); len(got) != 4 || got[0] != "sepal_length" {
		t.Errorf("screen.FeatureNames = %v", got)
	}
	if _, err := b.Float("screen_confidence"); err != nil {
		t.Errorf("screen_confidence: %v", err)
	}
	label, err := screen.Predict([][]float64{{5.1, 3.5, 1.4, 0.2}})
	if err != nil {
		t.Fatal(err)
	}
	if label[0] != 0 {
		t.Errorf("setosa predicted as %g", label[0])
	}
}
