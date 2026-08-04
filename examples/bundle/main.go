// Command bundle is a runnable example of a model that is not one estimator.
//
// The artifact here holds two Iris classifiers and the thresholds that turn them
// into a cascade: a cheap 15-tree screen, and a slower 25-tree confirm model
// consulted only when the screen is unsure. The thresholds were tuned for
// specificity, so they are as much a fitted parameter as any split in a tree —
// and the reason for shipping them inside the bundle is that the alternative,
// hand-written constants beside the model, drifts the first time the model is
// retrained and nothing fails loudly when it does.
//
// The whole bundle is compiled into the binary by go-ml-gen (see
// examples/bundle/models), so nothing is read at runtime. goml.LoadBundleFile
// and goml.LoadBundleBytes give the same *goml.Bundle from JSON when you would
// rather load it.
//
//	go run ./examples/bundle
package main

import (
	"fmt"
	"log"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/examples/bundle/models"
)

// cascade is everything the decision needs, read out of the bundle once at
// startup. Reading it here rather than at each call site means a bundle missing
// a key fails immediately and loudly, instead of silently predicting with a zero
// threshold somewhere deep in a request.
type cascade struct {
	screen, confirm  goml.Classifier
	assembler        *goml.Assembler
	screenConfidence float64 // below this, ask the confirm model
	confirmPositive  float64 // confirm's probability needed to call it positive
	positiveClass    float64
	tunedFor         string
}

func newCascade(b *goml.Bundle) (*cascade, error) {
	c := &cascade{}
	var err error
	if c.screen, err = b.Classifier("screen"); err != nil {
		return nil, err
	}
	if c.confirm, err = b.Classifier("confirm"); err != nil {
		return nil, err
	}
	if c.assembler, err = goml.NewAssembler(c.screen); err != nil {
		return nil, err
	}
	if c.screenConfidence, err = b.Float("screen_confidence"); err != nil {
		return nil, err
	}
	if c.confirmPositive, err = b.Float("confirm_positive"); err != nil {
		return nil, err
	}
	if c.positiveClass, err = b.Float("positive_class"); err != nil {
		return nil, err
	}
	if c.tunedFor, err = b.String("tuned_for"); err != nil {
		return nil, err
	}
	return c, nil
}

// decision is what the cascade concluded and how it got there.
type decision struct {
	ScreenLabel float64 // the screen's most probable class
	ScreenProba float64 // and its probability
	Escalated   bool    // the screen was not confident enough on its own
	ConfirmP    float64 // the confirm model's probability of the positive class
	Positive    bool    // ... and whether that cleared the tuned threshold
}

// classify runs the cascade over one sample.
func (c *cascade) classify(values map[string]float64) (decision, error) {
	row, err := c.assembler.Row(values) // by name: order cannot go wrong
	if err != nil {
		return decision{}, err
	}
	X := [][]float64{row}

	proba, err := c.screen.PredictProba(X)
	if err != nil {
		return decision{}, err
	}
	label, top := best(c.screen.Classes(), proba[0])
	d := decision{ScreenLabel: label, ScreenProba: top}
	if top >= c.screenConfidence {
		return d, nil // confident enough; the slower model is not consulted
	}

	confirmProba, err := c.confirm.PredictProba(X)
	if err != nil {
		return decision{}, err
	}
	d.Escalated = true
	d.ConfirmP = probaOf(c.confirm.Classes(), confirmProba[0], c.positiveClass)
	d.Positive = d.ConfirmP >= c.confirmPositive
	return d, nil
}

// narrate describes a decision against the thresholds that produced it.
func (c *cascade) narrate(d decision) string {
	if !d.Escalated {
		return fmt.Sprintf("screen  class=%g p=%.3f (>= %.2f, decided here)",
			d.ScreenLabel, d.ScreenProba, c.screenConfidence)
	}
	verdict := "negative"
	if d.Positive {
		verdict = "positive"
	}
	return fmt.Sprintf("screen  class=%g p=%.3f (< %.2f, escalated)\n          "+
		"confirm p(%g)=%.3f -> %s (threshold %.2f)",
		d.ScreenLabel, d.ScreenProba, c.screenConfidence,
		c.positiveClass, d.ConfirmP, verdict, c.confirmPositive)
}

func main() {
	c, err := newCascade(models.IrisCascade)
	if err != nil {
		log.Fatalf("bundle: %v", err)
	}

	fmt.Printf("bundle models:   %v\n", models.IrisCascade.Names())
	fmt.Printf("bundle metadata: %v (tuned for %s)\n", models.IrisCascade.MetaKeys(), c.tunedFor)
	fmt.Printf("features:        %v\n\n", c.assembler.Names())

	samples := []map[string]float64{
		{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
		{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
		{"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.6},
		{"sepal_length": 5.9, "sepal_width": 3.0, "petal_width": 1.8}, // petal_length not measured
	}
	for _, s := range samples {
		d, err := c.classify(s)
		if err != nil {
			log.Fatalf("bundle: %v", err)
		}
		fmt.Printf("%v\n  %s\n", s, c.narrate(d))
	}

	// A key the bundle does not carry is an error, not a zero — the failure
	// mode that hand-written metadata beside a model produces silently.
	if _, err := models.IrisCascade.Float("screen_confidence_v2"); err != nil {
		fmt.Printf("\nmissing metadata is loud: %v\n", err)
	}
}

// best returns the most probable class label and its probability.
func best(classes, proba []float64) (float64, float64) {
	bi := 0
	for i := range proba {
		if proba[i] > proba[bi] {
			bi = i
		}
	}
	return classes[bi], proba[bi]
}

// probaOf returns the probability of one particular class label.
func probaOf(classes, proba []float64, label float64) float64 {
	for i, c := range classes {
		if c == label {
			return proba[i]
		}
	}
	return 0
}
