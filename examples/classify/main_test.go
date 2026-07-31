package main

import (
	"errors"
	"math"
	"testing"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/examples/classify/models"
)

func TestExampleModelsPredict(t *testing.T) {
	cases := []struct {
		name              string
		clf               goml.Classifier
		features, classes int
		samples           [][]float64
	}{
		{"iris", models.Iris, 4, 3, [][]float64{
			{5.1, 3.5, 1.4, 0.2},
			{6.7, 3.0, 5.2, 2.3},
			{5.9, 3.0, math.NaN(), 1.8}, // missing feature must not error
		}},
		{"extratrees_balanced", models.ExtraTreesBalanced, 10, 3, [][]float64{
			{0.79, -2.55, 2.26, 0.23, 0.54, 0.15, 0.08, -0.68, 0.28, -2.4},
			{-1.05, 2.53, -0.31, 2.38, 0.12, 1.68, -0.57, 1.28, -0.89, -0.82},
			{math.NaN(), 2.53, -0.31, 2.38, math.NaN(), 1.68, -0.57, 1.28, -0.89, -0.82},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.clf.NFeatures() != tc.features || len(tc.clf.Classes()) != tc.classes {
				t.Fatalf("unexpected model shape: %d features, %d classes",
					tc.clf.NFeatures(), len(tc.clf.Classes()))
			}
			proba, err := tc.clf.PredictProba(tc.samples)
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
			if _, err := tc.clf.Predict(tc.samples); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestExampleWrongFeatureCount(t *testing.T) {
	if _, err := models.Iris.PredictProba([][]float64{{1, 2}}); !errors.Is(err, goml.ErrNumFeatures) {
		t.Errorf("err = %v, want ErrNumFeatures", err)
	}
	if _, err := models.ExtraTreesBalanced.PredictProba([][]float64{{1, 2}}); !errors.Is(err, goml.ErrNumFeatures) {
		t.Errorf("err = %v, want ErrNumFeatures", err)
	}
}
