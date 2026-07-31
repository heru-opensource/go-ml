package tree_test

import (
	"encoding/json"
	"fmt"
	"log"
	"math"

	"github.com/heru-opensource/go-ml/tree"
)

// stump returns a two-leaf tree over one feature: class 0 when feature 0 is
// <= 0.5, class 1 above it. A missing (NaN) value routes right, because
// MissingLeft is false at the root.
//
// Real trees come from scikit-learn through tools/sklexport; this one is small
// enough to read.
func stump() *tree.Tree {
	return &tree.Tree{
		Left:        []int32{1, -1, -1},
		Right:       []int32{2, -1, -1},
		Feature:     []int32{0, -2, -2},
		Threshold:   []float64{0.5, -2, -2},
		MissingLeft: []bool{false, false, false},
		Value:       []float64{0.5, 0.5, 1, 0, 0, 1},
		ValueWidth:  2,
	}
}

// ExampleTree_Apply finds the leaf a sample falls into. Node 0 is the root, so
// the ids here are the two leaves of the stump.
func ExampleTree_Apply() {
	t := stump()
	fmt.Println(t.Apply([]float64{0.2}), t.Apply([]float64{0.9}))
	// Output:
	// 1 2
}

// ExampleTree_Decide reads the class distribution stored at the leaf a sample
// reaches — what a DecisionTreeClassifier's predict_proba returns for it.
func ExampleTree_Decide() {
	t := stump()
	out := make([]float64, t.ValueWidth)

	t.Decide([]float64{0.2}, out)
	fmt.Printf("%.1f\n", out)

	// A missing feature does not compare: it follows the node's MissingLeft
	// flag, which is false here, so it routes right.
	t.Decide([]float64{math.NaN()}, out)
	fmt.Printf("%.1f\n", out)
	// Output:
	// [1.0 0.0]
	// [0.0 1.0]
}

// ExampleTree_AddTo accumulates leaf distributions across trees without
// allocating. It is the primitive the ensemble package's forests are built on:
// sum, then divide by the number of trees.
func ExampleTree_AddTo() {
	trees := []*tree.Tree{stump(), stump()}
	acc := make([]float64, 2)
	for _, t := range trees {
		t.AddTo([]float64{0.9}, acc)
	}
	for i := range acc {
		acc[i] /= float64(len(trees))
	}
	fmt.Printf("%.1f\n", acc)
	// Output:
	// [0.0 1.0]
}

// ExampleJSON_Build decodes the serialized form of a tree, as it appears inside
// a go-ml/v1 export, and validates it. Note the "Infinity" threshold: JSON
// cannot spell ±Inf, and scikit-learn uses an infinite threshold for a split
// that only separates missing values, so the export encodes it as a string.
func ExampleJSON_Build() {
	const serialized = `{
	  "node_count": 3, "value_width": 2,
	  "left": [1, -1, -1], "right": [2, -1, -1], "feature": [0, -2, -2],
	  "threshold": ["Infinity", -2, -2], "missing_left": [false, false, false],
	  "value": [0.5, 0.5, 1, 0, 0, 1]
	}`

	var j tree.JSON
	if err := json.Unmarshal([]byte(serialized), &j); err != nil {
		log.Fatal(err)
	}
	t, err := j.Build()
	if err != nil {
		log.Fatal(err)
	}

	// Every finite value is <= +Inf, so only a missing feature reaches the
	// right leaf.
	fmt.Println(t.Apply([]float64{1e308}), t.Apply([]float64{math.NaN()}))
	// Output:
	// 1 2
}
