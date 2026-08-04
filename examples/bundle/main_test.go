package main

import (
	"strings"
	"testing"

	"github.com/heru-opensource/go-ml/examples/bundle/models"
)

func newTestCascade(t *testing.T) *cascade {
	t.Helper()
	c, err := newCascade(models.IrisCascade)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestCascadeReadsItsThresholds is the property the bundle buys: every tuned
// number the decision needs comes from the artifact, and a bundle missing one
// fails at startup instead of predicting with a zero.
func TestCascadeReadsItsThresholds(t *testing.T) {
	c := newTestCascade(t)
	if c.screenConfidence <= 0 || c.screenConfidence > 1 {
		t.Errorf("screen_confidence = %v", c.screenConfidence)
	}
	if c.confirmPositive <= 0 || c.confirmPositive > 1 {
		t.Errorf("confirm_positive = %v", c.confirmPositive)
	}
	if c.tunedFor == "" {
		t.Error("tuned_for is empty")
	}
	if c.screen.Type() != "RandomForestClassifier" || c.confirm.Type() != "ExtraTreesClassifier" {
		t.Errorf("models are %s and %s", c.screen.Type(), c.confirm.Type())
	}
}

func TestCascadeShortCircuitsWhenConfident(t *testing.T) {
	c := newTestCascade(t)
	d, err := c.classify(map[string]float64{
		"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if d.Escalated {
		t.Errorf("an obvious setosa should not reach the confirm model; got %+v", d)
	}
	if d.ScreenProba < c.screenConfidence {
		t.Errorf("screen p=%v is below the threshold it supposedly cleared", d.ScreenProba)
	}
	if !strings.Contains(c.narrate(d), "decided here") {
		t.Errorf("narration = %q", c.narrate(d))
	}
}

func TestCascadeEscalatesWhenUnsure(t *testing.T) {
	c := newTestCascade(t)
	d, err := c.classify(map[string]float64{
		"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.6,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !d.Escalated {
		t.Fatalf("a borderline sample should reach the confirm model; got %+v", d)
	}
	if d.Positive != (d.ConfirmP >= c.confirmPositive) {
		t.Errorf("verdict %v disagrees with p=%v against threshold %v",
			d.Positive, d.ConfirmP, c.confirmPositive)
	}
}

// TestCascadeRejectsUnknownFeature covers the assembler wiring: the cascade
// takes named values, so a typo cannot become a misplaced number.
func TestCascadeRejectsUnknownFeature(t *testing.T) {
	c := newTestCascade(t)
	if _, err := c.classify(map[string]float64{"petal_len": 1.4}); err == nil {
		t.Error("expected an error for an unknown feature name")
	}
}
