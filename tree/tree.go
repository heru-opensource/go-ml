// Package tree implements binary decision trees whose evaluation is bit-for-bit
// compatible with scikit-learn's sklearn.tree._tree.Tree.
//
// A [Tree] is stored as flat, parallel, node-indexed arrays, exactly the layout
// scikit-learn serializes. Node 0 is the root. Evaluation ([Tree.Apply])
// reproduces scikit-learn's _apply_dense routine precisely, including two
// details that matter for exact agreement:
//
//   - float32 narrowing. scikit-learn converts the input matrix X to float32
//     before traversal, so a feature value is compared as
//     float64(float32(x)) <= threshold (the threshold itself stays float64).
//   - missing values. A NaN feature does not compare; it routes to the left
//     child when the node's MissingLeft flag is set, otherwise to the right.
//
// The package has no dependency on the rest of go-ml and can be used on its own
// to evaluate a single decision tree.
package tree

import (
	"fmt"
	"math"

	"github.com/heru-public/go-ml/internal/jsonx"
)

// Tree is a fitted binary decision tree in scikit-learn's flat-array form.
//
// All slices are indexed by node id and have the same length (the node count).
// For an internal node i, Left[i] and Right[i] are child node ids; a sample
// routes to Left[i] when feature Feature[i] is missing and MissingLeft[i] is
// true, or when the (float32-narrowed) feature value is <= Threshold[i];
// otherwise it routes to Right[i]. A leaf has Left[i] < 0 (and Right[i] < 0)
// and carries a length-ValueWidth output vector at Value[i*ValueWidth:].
//
// The exported fields make a Tree trivially constructible by generated code;
// use [JSON.Build] to construct one from the export format with validation.
type Tree struct {
	Left        []int32   // left child id, or <0 for a leaf
	Right       []int32   // right child id, or <0 for a leaf
	Feature     []int32   // split feature index (unused at leaves)
	Threshold   []float64 // split threshold (unused at leaves)
	MissingLeft []bool    // a missing (NaN) feature routes left when true
	Value       []float64 // node output vectors, flattened row-major
	ValueWidth  int       // length of each node's output vector
}

// Apply returns the id of the leaf node that x falls into.
//
// It reproduces scikit-learn's _apply_dense exactly: the feature value is first
// narrowed to float32 (matching scikit-learn's conversion of X), a NaN routes
// by MissingLeft, and otherwise the value—widened back to float64—is compared
// with <= against the float64 threshold. x must have at least as many elements
// as the largest split feature index in the tree.
func (t *Tree) Apply(x []float64) int32 {
	n := int32(0)
	for t.Left[n] >= 0 {
		// Narrow to float32 and widen back, exactly as scikit-learn does when
		// it casts X to float32 before reading the feature.
		v := float64(float32(x[t.Feature[n]]))
		switch {
		case math.IsNaN(v):
			if t.MissingLeft[n] {
				n = t.Left[n]
			} else {
				n = t.Right[n]
			}
		case v <= t.Threshold[n]:
			n = t.Left[n]
		default:
			n = t.Right[n]
		}
	}
	return n
}

// ValueAt returns the output vector stored at node id n (length ValueWidth).
// The returned slice aliases the tree's storage; do not mutate it.
func (t *Tree) ValueAt(n int32) []float64 {
	off := int(n) * t.ValueWidth
	return t.Value[off : off+t.ValueWidth]
}

// Decide writes the output vector of the leaf reached by x into out, which must
// have length ValueWidth. For a classification tree this is the leaf's class
// probability distribution.
func (t *Tree) Decide(x, out []float64) {
	copy(out, t.ValueAt(t.Apply(x)))
}

// AddTo adds the output vector of the leaf reached by x into acc element-wise.
// acc must have length ValueWidth. It is the hot primitive ensembles use to
// accumulate per-tree contributions without allocating.
func (t *Tree) AddTo(x, acc []float64) {
	v := t.ValueAt(t.Apply(x))
	for i := range acc {
		acc[i] += v[i]
	}
}

// JSON is the serialized form of a Tree in the go-ml/v1 export format. Use
// [JSON.Build] to validate it and produce a [Tree].
type JSON struct {
	NodeCount   int          `json:"node_count"`
	ValueWidth  int          `json:"value_width"`
	Left        []int32      `json:"left"`
	Right       []int32      `json:"right"`
	Feature     []int32      `json:"feature"`
	Threshold   jsonx.Floats `json:"threshold"`
	MissingLeft []bool       `json:"missing_left"`
	Value       jsonx.Floats `json:"value"`
}

// Build validates the serialized tree and returns an evaluatable [Tree].
func (j *JSON) Build() (*Tree, error) {
	n := j.NodeCount
	if n <= 0 {
		return nil, fmt.Errorf("tree: node_count must be positive, got %d", n)
	}
	if j.ValueWidth <= 0 {
		return nil, fmt.Errorf("tree: value_width must be positive, got %d", j.ValueWidth)
	}
	if len(j.Left) != n || len(j.Right) != n || len(j.Feature) != n ||
		len(j.Threshold) != n || len(j.MissingLeft) != n {
		return nil, fmt.Errorf("tree: node arrays disagree with node_count %d "+
			"(left=%d right=%d feature=%d threshold=%d missing_left=%d)",
			n, len(j.Left), len(j.Right), len(j.Feature), len(j.Threshold), len(j.MissingLeft))
	}
	if len(j.Value) != n*j.ValueWidth {
		return nil, fmt.Errorf("tree: value has %d elements, want node_count*value_width = %d",
			len(j.Value), n*j.ValueWidth)
	}
	for i := 0; i < n; i++ {
		l, r := j.Left[i], j.Right[i]
		if l >= int32(n) || r >= int32(n) {
			return nil, fmt.Errorf("tree: node %d child out of range (left=%d right=%d node_count=%d)", i, l, r, n)
		}
		// A node is either a leaf (both children < 0) or internal (both valid).
		isLeaf := l < 0
		if isLeaf != (r < 0) {
			return nil, fmt.Errorf("tree: node %d has exactly one leaf child (left=%d right=%d)", i, l, r)
		}
		if !isLeaf && j.Feature[i] < 0 {
			return nil, fmt.Errorf("tree: internal node %d has negative feature index %d", i, j.Feature[i])
		}
	}
	return &Tree{
		Left:        j.Left,
		Right:       j.Right,
		Feature:     j.Feature,
		Threshold:   []float64(j.Threshold),
		MissingLeft: j.MissingLeft,
		Value:       []float64(j.Value),
		ValueWidth:  j.ValueWidth,
	}, nil
}
