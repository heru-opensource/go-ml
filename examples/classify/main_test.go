package main

import (
	"errors"
	"math"
	"testing"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/examples/classify/models"
)

func TestExampleModelPredicts(t *testing.T) {
	clf := models.Iris
	if clf.NFeatures() != 4 || len(clf.Classes()) != 3 {
		t.Fatalf("unexpected model shape: %d features, %d classes", clf.NFeatures(), len(clf.Classes()))
	}

	samples := [][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{6.7, 3.0, 5.2, 2.3},
		{5.9, 3.0, math.NaN(), 1.8}, // missing feature must not error
	}
	proba, err := clf.PredictProba(samples)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range proba {
		var sum float64
		for _, v := range row {
			sum += v
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("sample %d probabilities sum to %v, want 1", i, sum)
		}
	}
	if _, err := clf.Predict(samples); err != nil {
		t.Fatal(err)
	}
}

func TestExampleWrongFeatureCount(t *testing.T) {
	if _, err := models.Iris.PredictProba([][]float64{{1, 2}}); !errors.Is(err, goml.ErrNumFeatures) {
		t.Errorf("err = %v, want ErrNumFeatures", err)
	}
}
