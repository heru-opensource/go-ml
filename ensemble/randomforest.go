// Package ensemble implements tree-ensemble models. It currently provides
// [RandomForestClassifier], a static, parallel reimplementation of
// scikit-learn's sklearn.ensemble.RandomForestClassifier prediction path.
//
// Importing this package registers its model types with the parent goml
// package, so they can be loaded through goml.Load:
//
//	import _ "github.com/heru-public/go-ml/ensemble"
package ensemble

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/internal/jsonx"
	"github.com/heru-public/go-ml/tree"
)

// TypeRandomForestClassifier is the export "type" value for a random forest
// classifier, and the value returned by [RandomForestClassifier.Type].
const TypeRandomForestClassifier = "RandomForestClassifier"

// parallelMinWork is the rows×trees product below which prediction runs on a
// single goroutine. Spawning workers for a handful of shallow trees costs more
// than it saves, and the single-goroutine path is bit-for-bit identical to
// scikit-learn; above the threshold the work is split across goroutines.
const parallelMinWork = 1 << 15

func init() {
	goml.Register(TypeRandomForestClassifier, func(raw json.RawMessage) (goml.Model, error) {
		return decodeRandomForest(raw)
	})
}

// RandomForestClassifier reproduces scikit-learn's RandomForestClassifier
// prediction: each decision tree yields the class distribution of the leaf a
// sample reaches, and the forest's probability is the mean of those
// distributions over all trees (see [RandomForestClassifier.PredictProba]).
// Predictions are computed concurrently across goroutines for large batches or
// large forests.
//
// A RandomForestClassifier is safe for concurrent use by multiple goroutines.
type RandomForestClassifier struct {
	nFeatures int
	classes   []float64
	trees     []*tree.Tree
	workers   int // <=0 means GOMAXPROCS
}

// Option configures a model at construction time.
type Option func(*options)

type options struct {
	workers int
}

// WithWorkers sets the maximum number of goroutines used per prediction call.
// The zero value (or any n <= 0) means runtime.GOMAXPROCS. WithWorkers(1)
// forces sequential prediction, which is bit-for-bit identical to scikit-learn.
func WithWorkers(n int) Option {
	return func(o *options) { o.workers = n }
}

func applyOptions(opts []Option) options {
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// NewRandomForestClassifier constructs a forest from its parts. classes are the
// labels in column order; each tree's Value vectors must already be L1-
// normalized leaf class distributions of width len(classes). This is the
// constructor that statically generated code (see cmd/go-ml-gen) calls; most
// callers load a model instead with [LoadRandomForestClassifier] or goml.Load.
func NewRandomForestClassifier(nFeatures int, classes []float64, trees []*tree.Tree, opts ...Option) (*RandomForestClassifier, error) {
	if nFeatures <= 0 {
		return nil, fmt.Errorf("ensemble: n_features must be positive, got %d", nFeatures)
	}
	if len(classes) < 2 {
		return nil, fmt.Errorf("ensemble: need at least 2 classes, got %d", len(classes))
	}
	if len(trees) == 0 {
		return nil, fmt.Errorf("ensemble: forest has no trees")
	}
	for i, t := range trees {
		if t == nil {
			return nil, fmt.Errorf("ensemble: tree %d is nil", i)
		}
		if t.ValueWidth != len(classes) {
			return nil, fmt.Errorf("ensemble: tree %d value width %d != n_classes %d", i, t.ValueWidth, len(classes))
		}
	}
	o := applyOptions(opts)
	return &RandomForestClassifier{
		nFeatures: nFeatures,
		classes:   append([]float64(nil), classes...),
		trees:     trees,
		workers:   o.workers,
	}, nil
}

// Type returns TypeRandomForestClassifier.
func (rf *RandomForestClassifier) Type() string { return TypeRandomForestClassifier }

// NFeatures returns the number of input features each sample must have.
func (rf *RandomForestClassifier) NFeatures() int { return rf.nFeatures }

// NTrees returns the number of decision trees in the forest.
func (rf *RandomForestClassifier) NTrees() int { return len(rf.trees) }

// Classes returns a copy of the class labels in PredictProba column order.
func (rf *RandomForestClassifier) Classes() []float64 {
	return append([]float64(nil), rf.classes...)
}

// PredictProba returns the mean per-class probability over all trees for each
// sample in X. The result has one row per input sample and len(Classes)
// columns, aligned with Classes. A missing feature is encoded as math.NaN.
func (rf *RandomForestClassifier) PredictProba(X [][]float64) ([][]float64, error) {
	if err := rf.checkX(X); err != nil {
		return nil, err
	}
	nc := len(rf.classes)
	out := make([][]float64, len(X))
	for i := range out {
		out[i] = make([]float64, nc)
	}
	rf.accumulate(X, out)
	inv := 1.0 / float64(len(rf.trees))
	for i := range out {
		for c := range out[i] {
			out[i][c] *= inv
		}
	}
	return out, nil
}

// Predict returns the most probable class label for each sample in X. Ties are
// resolved toward the lowest column index, matching numpy.argmax.
func (rf *RandomForestClassifier) Predict(X [][]float64) ([]float64, error) {
	proba, err := rf.PredictProba(X)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(proba))
	for i, p := range proba {
		out[i] = rf.classes[argmax(p)]
	}
	return out, nil
}

// PredictProbaRow is PredictProba for a single sample.
func (rf *RandomForestClassifier) PredictProbaRow(x []float64) ([]float64, error) {
	proba, err := rf.PredictProba([][]float64{x})
	if err != nil {
		return nil, err
	}
	return proba[0], nil
}

// PredictRow is Predict for a single sample.
func (rf *RandomForestClassifier) PredictRow(x []float64) (float64, error) {
	proba, err := rf.PredictProbaRow(x)
	if err != nil {
		return 0, err
	}
	return rf.classes[argmax(proba)], nil
}

func (rf *RandomForestClassifier) checkX(X [][]float64) error {
	for i, x := range X {
		if len(x) != rf.nFeatures {
			return fmt.Errorf("%w: sample %d has %d, want %d", goml.ErrNumFeatures, i, len(x), rf.nFeatures)
		}
	}
	return nil
}

// accumulate fills out[i] with the sum over all trees of the leaf class
// distribution reached by X[i] (the division by NTrees happens in the caller).
func (rf *RandomForestClassifier) accumulate(X [][]float64, out [][]float64) {
	w := rf.numWorkers()
	work := int64(len(X)) * int64(len(rf.trees))
	switch {
	case w <= 1 || work < parallelMinWork:
		rf.sumTrees(X, out, 0, len(rf.trees))
	case len(X) >= w:
		rf.accumulateByRows(X, out, w)
	default:
		rf.accumulateByTrees(X, out, w)
	}
}

// sumTrees adds the contributions of trees[t0:t1] for every sample into out.
func (rf *RandomForestClassifier) sumTrees(X [][]float64, out [][]float64, t0, t1 int) {
	for i, x := range X {
		acc := out[i]
		for _, t := range rf.trees[t0:t1] {
			t.AddTo(x, acc)
		}
	}
}

// accumulateByRows splits the samples across w goroutines. Each output row is
// written by exactly one goroutine and sums trees in order, so the result is
// bit-for-bit identical to the sequential path.
func (rf *RandomForestClassifier) accumulateByRows(X [][]float64, out [][]float64, w int) {
	var wg sync.WaitGroup
	chunk := (len(X) + w - 1) / w
	for start := 0; start < len(X); start += chunk {
		end := min(start+chunk, len(X))
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			rf.sumTrees(X[s:e], out[s:e], 0, len(rf.trees))
		}(start, end)
	}
	wg.Wait()
}

// accumulateByTrees splits the trees across w goroutines, used when there are
// fewer samples than workers (e.g. single-row prediction over a huge forest).
// Each goroutine accumulates its tree range into a private buffer; the buffers
// are then summed into out in a fixed order, so the result is deterministic
// (and within floating-point rounding of the sequential path).
func (rf *RandomForestClassifier) accumulateByTrees(X [][]float64, out [][]float64, w int) {
	nc := len(rf.classes)
	nrows := len(X)
	chunk := (len(rf.trees) + w - 1) / w

	type part struct{ buf []float64 }
	var parts []part
	var wg sync.WaitGroup
	for start := 0; start < len(rf.trees); start += chunk {
		end := min(start+chunk, len(rf.trees))
		buf := make([]float64, nrows*nc)
		parts = append(parts, part{buf})
		wg.Add(1)
		go func(s, e int, buf []float64) {
			defer wg.Done()
			for i, x := range X {
				acc := buf[i*nc : (i+1)*nc]
				for _, t := range rf.trees[s:e] {
					t.AddTo(x, acc)
				}
			}
		}(start, end, buf)
	}
	wg.Wait()

	for _, p := range parts {
		for i := 0; i < nrows; i++ {
			src := p.buf[i*nc : (i+1)*nc]
			dst := out[i]
			for c := range dst {
				dst[c] += src[c]
			}
		}
	}
}

func (rf *RandomForestClassifier) numWorkers() int {
	if rf.workers > 0 {
		return rf.workers
	}
	if n := runtime.GOMAXPROCS(0); n > 1 {
		return n
	}
	return 1
}

func argmax(p []float64) int {
	best, bi := p[0], 0
	for i := 1; i < len(p); i++ {
		if p[i] > best {
			best, bi = p[i], i
		}
	}
	return bi
}

// --- serialization ---

type rfJSON struct {
	NFeatures int          `json:"n_features"`
	NOutputs  int          `json:"n_outputs"`
	Classes   jsonx.Floats `json:"classes"`
	Trees     []tree.JSON  `json:"trees"`
}

func decodeRandomForest(raw json.RawMessage) (*RandomForestClassifier, error) {
	var j rfJSON
	if err := json.Unmarshal(raw, &j); err != nil {
		return nil, err
	}
	if j.NOutputs != 0 && j.NOutputs != 1 {
		return nil, fmt.Errorf("ensemble: only single-output forests are supported, got n_outputs=%d", j.NOutputs)
	}
	trees := make([]*tree.Tree, len(j.Trees))
	for i := range j.Trees {
		t, err := j.Trees[i].Build()
		if err != nil {
			return nil, fmt.Errorf("tree %d: %w", i, err)
		}
		trees[i] = t
	}
	return NewRandomForestClassifier(j.NFeatures, []float64(j.Classes), trees)
}

// LoadRandomForestClassifier decodes a forest from a go-ml/v1 export envelope
// read from r, applying the given options. Unlike goml.Load it returns the
// concrete type, exposing helpers such as NTrees and the single-row methods.
func LoadRandomForestClassifier(data []byte, opts ...Option) (*RandomForestClassifier, error) {
	m, err := goml.LoadBytes(data)
	if err != nil {
		return nil, err
	}
	rf, ok := m.(*RandomForestClassifier)
	if !ok {
		return nil, fmt.Errorf("ensemble: export is a %s, not a RandomForestClassifier", m.Type())
	}
	if o := applyOptions(opts); o.workers != 0 {
		rf.workers = o.workers
	}
	return rf, nil
}
