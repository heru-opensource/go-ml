package tree

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/heru-opensource/go-ml/internal/jsonx"
)

// stump builds a 3-node tree: root splits on feature 0 at the given threshold,
// routing missing values left iff missingLeft; the leaves emit [1,0] and [0,1].
func stump(threshold float64, missingLeft bool) *Tree {
	return &Tree{
		Left:        []int32{1, -1, -1},
		Right:       []int32{2, -1, -1},
		Feature:     []int32{0, -2, -2},
		Threshold:   []float64{threshold, -2, -2},
		MissingLeft: []bool{missingLeft, false, false},
		Value:       []float64{0.5, 0.5, 1, 0, 0, 1},
		ValueWidth:  2,
	}
}

func TestApplyRouting(t *testing.T) {
	tr := stump(10, false)
	if leaf := tr.Apply([]float64{5}); leaf != 1 { // 5 <= 10 -> left
		t.Errorf("x=5 reached leaf %d, want 1 (left)", leaf)
	}
	if leaf := tr.Apply([]float64{10}); leaf != 1 { // 10 <= 10 -> left
		t.Errorf("x=10 reached leaf %d, want 1 (left)", leaf)
	}
	if leaf := tr.Apply([]float64{15}); leaf != 2 { // 15 > 10 -> right
		t.Errorf("x=15 reached leaf %d, want 2 (right)", leaf)
	}
}

func TestApplyMissingRouting(t *testing.T) {
	nan := math.NaN()
	if leaf := stump(10, true).Apply([]float64{nan}); leaf != 1 {
		t.Errorf("NaN with missingLeft=true reached leaf %d, want 1 (left)", leaf)
	}
	if leaf := stump(10, false).Apply([]float64{nan}); leaf != 2 {
		t.Errorf("NaN with missingLeft=false reached leaf %d, want 2 (right)", leaf)
	}
}

// TestApplyFloat32Cast pins down the subtle behavior that makes go-ml agree
// with scikit-learn: the feature value is narrowed to float32 before the
// comparison. With threshold 16777216.5 and x = 16777217 (=2^24+1, which has no
// float32 representation and rounds down to 2^24), the float32 cast makes the
// value compare <= threshold and route left, whereas a naive float64 compare
// would route right.
func TestApplyFloat32Cast(t *testing.T) {
	const threshold = 16777216.5
	const x = 16777217.0

	if x <= threshold { // sanity: naive float64 comparison routes right
		t.Fatal("test premise broken: x should be > threshold in float64")
	}
	if leaf := stump(threshold, false).Apply([]float64{x}); leaf != 1 {
		t.Errorf("float32 cast not applied: reached leaf %d, want 1 (left)", leaf)
	}
}

func TestDecideAndValueAt(t *testing.T) {
	tr := stump(10, false)
	out := make([]float64, tr.ValueWidth)
	tr.Decide([]float64{5}, out)
	if out[0] != 1 || out[1] != 0 {
		t.Errorf("Decide(left) = %v, want [1 0]", out)
	}
	acc := []float64{10, 20}
	tr.AddTo([]float64{15}, acc) // right leaf is [0,1]
	if acc[0] != 10 || acc[1] != 21 {
		t.Errorf("AddTo = %v, want [10 21]", acc)
	}
}

func TestBuildAndJSONRoundTrip(t *testing.T) {
	j := JSON{
		NodeCount: 3, ValueWidth: 2,
		Left:        []int32{1, -1, -1},
		Right:       []int32{2, -1, -1},
		Feature:     []int32{0, -2, -2},
		Threshold:   jsonx.Floats{math.Inf(1), -2, -2}, // pure missing-value split
		MissingLeft: []bool{false, false, false},
		Value:       jsonx.Floats{0.5, 0.5, 1, 0, 0, 1},
	}
	tr, err := j.Build()
	if err != nil {
		t.Fatal(err)
	}
	// With an infinite threshold every finite value routes left.
	if leaf := tr.Apply([]float64{1e300}); leaf != 1 {
		t.Errorf("inf-threshold routed %d, want 1 (left)", leaf)
	}

	// The serialized form must survive a JSON round trip (incl. the inf).
	blob, err := json.Marshal(map[string]any{
		"node_count": 3, "value_width": 2,
		"left": j.Left, "right": j.Right, "feature": j.Feature,
		"threshold": []any{"Infinity", -2, -2}, "missing_left": j.MissingLeft,
		"value": j.Value,
	})
	if err != nil {
		t.Fatal(err)
	}
	var j2 JSON
	if err := json.Unmarshal(blob, &j2); err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(j2.Threshold[0], 1) {
		t.Errorf("threshold[0] = %v, want +Inf", j2.Threshold[0])
	}
}

func TestBuildValidation(t *testing.T) {
	base := func() JSON {
		return JSON{
			NodeCount: 3, ValueWidth: 2,
			Left: []int32{1, -1, -1}, Right: []int32{2, -1, -1},
			Feature: []int32{0, -2, -2}, Threshold: jsonx.Floats{10, -2, -2},
			MissingLeft: []bool{false, false, false},
			Value:       jsonx.Floats{0.5, 0.5, 1, 0, 0, 1},
		}
	}
	tests := map[string]func(*JSON){
		"zero node count":    func(j *JSON) { j.NodeCount = 0 },
		"bad value width":    func(j *JSON) { j.ValueWidth = 0 },
		"short array":        func(j *JSON) { j.Left = []int32{1, -1} },
		"value len mismatch": func(j *JSON) { j.Value = jsonx.Floats{1, 2, 3} },
		"child out of range": func(j *JSON) { j.Right = []int32{99, -1, -1} },
		"half leaf":          func(j *JSON) { j.Left = []int32{1, -1, -1}; j.Right = []int32{-1, -1, -1} },
	}
	for name, mutate := range tests {
		j := base()
		mutate(&j)
		if _, err := j.Build(); err == nil {
			t.Errorf("%s: expected Build error, got nil", name)
		}
	}
	valid := base()
	if _, err := valid.Build(); err != nil {
		t.Errorf("valid tree failed to build: %v", err)
	}
}
