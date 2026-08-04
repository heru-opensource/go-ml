package ensemble

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sync"

	goml "github.com/heru-opensource/go-ml"
	"github.com/heru-opensource/go-ml/internal/jsonx"
	"github.com/heru-opensource/go-ml/tree"
)

// parallelMinWork is the rows×trees product below which prediction runs on a
// single goroutine. Spawning workers for a handful of shallow trees costs more
// than it saves, and the single-goroutine path is bit-for-bit identical to
// scikit-learn; above the threshold the work is split across goroutines.
const parallelMinWork = 1 << 15

// Forest is the interface every tree-ensemble classifier in this package
// satisfies: goml.Classifier plus the forest-specific helpers. Program against
// it to write code that serves a [RandomForestClassifier] and an
// [ExtraTreesClassifier] alike; see [LoadForest].
type Forest interface {
	goml.Classifier

	// NTrees returns the number of decision trees in the ensemble.
	NTrees() int

	// PredictProbaRow is PredictProba for a single sample.
	PredictProbaRow(x []float64) ([]float64, error)

	// PredictRow is Predict for a single sample.
	PredictRow(x []float64) (float64, error)
}

// Option configures a model at construction time.
type Option func(*options)

type options struct {
	workers      int
	featureNames []string
}

// WithWorkers sets the maximum number of goroutines used per prediction call.
// The zero value (or any n <= 0) means runtime.GOMAXPROCS. WithWorkers(1)
// forces sequential prediction, which is bit-for-bit identical to scikit-learn.
func WithWorkers(n int) Option {
	return func(o *options) { o.workers = n }
}

// WithFeatureNames attaches input feature names, in the model's own column
// order. It is how statically generated code carries scikit-learn's
// feature_names_in_ (see cmd/go-ml-gen); an export that has them supplies them
// on its own, and passing this to a loader overrides what the file said. The
// count must match n_features.
func WithFeatureNames(names []string) Option {
	return func(o *options) { o.featureNames = names }
}

func applyOptions(opts []Option) options {
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// forest is the shared implementation behind the package's [Forest] models.
// The exported types embed it, which is why they have identical method sets:
// scikit-learn predicts identically from a random forest and from extremely
// randomized trees, so only the estimator name and the trees differ.
//
// A forest is safe for concurrent use by multiple goroutines.
type forest struct {
	nFeatures    int
	classes      []float64
	featureNames []string // nil when the export carried none
	trees        []*tree.Tree
	workers      int // <=0 means GOMAXPROCS
}

// newForest validates the parts of an ensemble and assembles them. classes are
// the labels in column order; each tree's Value vectors must already be L1-
// normalized leaf class distributions of width len(classes).
func newForest(nFeatures int, classes []float64, trees []*tree.Tree, opts []Option) (forest, error) {
	if nFeatures <= 0 {
		return forest{}, fmt.Errorf("ensemble: n_features must be positive, got %d", nFeatures)
	}
	if len(classes) < 2 {
		return forest{}, fmt.Errorf("ensemble: need at least 2 classes, got %d", len(classes))
	}
	if len(trees) == 0 {
		return forest{}, fmt.Errorf("ensemble: forest has no trees")
	}
	for i, t := range trees {
		if t == nil {
			return forest{}, fmt.Errorf("ensemble: tree %d is nil", i)
		}
		if t.ValueWidth != len(classes) {
			return forest{}, fmt.Errorf("ensemble: tree %d value width %d != n_classes %d", i, t.ValueWidth, len(classes))
		}
	}
	o := applyOptions(opts)
	if n := len(o.featureNames); n > 0 && n != nFeatures {
		return forest{}, fmt.Errorf("ensemble: %d feature names for %d features", n, nFeatures)
	}
	return forest{
		nFeatures:    nFeatures,
		classes:      append([]float64(nil), classes...),
		featureNames: append([]string(nil), o.featureNames...),
		trees:        trees,
		workers:      o.workers,
	}, nil
}

// NFeatures returns the number of input features each sample must have.
func (f *forest) NFeatures() int { return f.nFeatures }

// FeatureNames returns a copy of the input feature names in column order, or
// nil when the export carried none. See [goml.Model] and [goml.Assembler].
func (f *forest) FeatureNames() []string {
	return append([]string(nil), f.featureNames...)
}

// NTrees returns the number of decision trees in the ensemble.
func (f *forest) NTrees() int { return len(f.trees) }

// Classes returns a copy of the class labels in PredictProba column order.
func (f *forest) Classes() []float64 {
	return append([]float64(nil), f.classes...)
}

// PredictProba returns the mean per-class probability over all trees for each
// sample in X. The result has one row per input sample and len(Classes)
// columns, aligned with Classes. A missing feature is encoded as math.NaN.
func (f *forest) PredictProba(X [][]float64) ([][]float64, error) {
	if err := f.checkX(X); err != nil {
		return nil, err
	}
	nc := len(f.classes)
	out := make([][]float64, len(X))
	for i := range out {
		out[i] = make([]float64, nc)
	}
	f.accumulate(X, out)
	inv := 1.0 / float64(len(f.trees))
	for i := range out {
		for c := range out[i] {
			out[i][c] *= inv
		}
	}
	return out, nil
}

// Predict returns the most probable class label for each sample in X. Ties are
// resolved toward the lowest column index, matching numpy.argmax.
func (f *forest) Predict(X [][]float64) ([]float64, error) {
	proba, err := f.PredictProba(X)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(proba))
	for i, p := range proba {
		out[i] = f.classes[argmax(p)]
	}
	return out, nil
}

// PredictProbaRow is PredictProba for a single sample.
func (f *forest) PredictProbaRow(x []float64) ([]float64, error) {
	proba, err := f.PredictProba([][]float64{x})
	if err != nil {
		return nil, err
	}
	return proba[0], nil
}

// PredictRow is Predict for a single sample.
func (f *forest) PredictRow(x []float64) (float64, error) {
	proba, err := f.PredictProbaRow(x)
	if err != nil {
		return 0, err
	}
	return f.classes[argmax(proba)], nil
}

func (f *forest) checkX(X [][]float64) error {
	for i, x := range X {
		if len(x) != f.nFeatures {
			return fmt.Errorf("%w: sample %d has %d, want %d", goml.ErrNumFeatures, i, len(x), f.nFeatures)
		}
	}
	return nil
}

// accumulate fills out[i] with the sum over all trees of the leaf class
// distribution reached by X[i] (the division by NTrees happens in the caller).
func (f *forest) accumulate(X [][]float64, out [][]float64) {
	w := f.numWorkers()
	work := int64(len(X)) * int64(len(f.trees))
	switch {
	case w <= 1 || work < parallelMinWork:
		f.sumTrees(X, out, 0, len(f.trees))
	case len(X) >= w:
		f.accumulateByRows(X, out, w)
	default:
		f.accumulateByTrees(X, out, w)
	}
}

// sumTrees adds the contributions of trees[t0:t1] for every sample into out.
func (f *forest) sumTrees(X [][]float64, out [][]float64, t0, t1 int) {
	for i, x := range X {
		acc := out[i]
		for _, t := range f.trees[t0:t1] {
			t.AddTo(x, acc)
		}
	}
}

// accumulateByRows splits the samples across w goroutines. Each output row is
// written by exactly one goroutine and sums trees in order, so the result is
// bit-for-bit identical to the sequential path.
func (f *forest) accumulateByRows(X [][]float64, out [][]float64, w int) {
	var wg sync.WaitGroup
	chunk := (len(X) + w - 1) / w
	for start := 0; start < len(X); start += chunk {
		end := min(start+chunk, len(X))
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			f.sumTrees(X[s:e], out[s:e], 0, len(f.trees))
		}(start, end)
	}
	wg.Wait()
}

// accumulateByTrees splits the trees across w goroutines, used when there are
// fewer samples than workers (e.g. single-row prediction over a huge forest).
// Each goroutine accumulates its tree range into a private buffer; the buffers
// are then summed into out in a fixed order, so the result is deterministic
// (and within floating-point rounding of the sequential path).
func (f *forest) accumulateByTrees(X [][]float64, out [][]float64, w int) {
	nc := len(f.classes)
	nrows := len(X)
	chunk := (len(f.trees) + w - 1) / w

	type part struct{ buf []float64 }
	var parts []part
	var wg sync.WaitGroup
	for start := 0; start < len(f.trees); start += chunk {
		end := min(start+chunk, len(f.trees))
		buf := make([]float64, nrows*nc)
		parts = append(parts, part{buf})
		wg.Add(1)
		go func(s, e int, buf []float64) {
			defer wg.Done()
			for i, x := range X {
				acc := buf[i*nc : (i+1)*nc]
				for _, t := range f.trees[s:e] {
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

func (f *forest) numWorkers() int {
	if f.workers > 0 {
		return f.workers
	}
	if n := runtime.GOMAXPROCS(0); n > 1 {
		return n
	}
	return 1
}

// setWorkers applies [WithWorkers] to an already-decoded model; the loaders use
// it because a registered goml decoder takes no options.
func (f *forest) setWorkers(n int) { f.workers = n }

// setFeatureNames applies [WithFeatureNames] to an already-decoded model,
// replacing whatever the export carried.
func (f *forest) setFeatureNames(names []string) error {
	if len(names) != f.nFeatures {
		return fmt.Errorf("ensemble: %d feature names for %d features", len(names), f.nFeatures)
	}
	f.featureNames = append([]string(nil), names...)
	return nil
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

// forestJSON is the "model" object of a go-ml/v1 export for either forest type.
type forestJSON struct {
	NFeatures    int          `json:"n_features"`
	NOutputs     int          `json:"n_outputs"`
	Classes      jsonx.Floats `json:"classes"`
	FeatureNames []string     `json:"feature_names"` // absent unless fitted on a named frame
	Trees        []tree.JSON  `json:"trees"`
}

func decodeForest(raw json.RawMessage) (forest, error) {
	var j forestJSON
	if err := json.Unmarshal(raw, &j); err != nil {
		return forest{}, err
	}
	if j.NOutputs != 0 && j.NOutputs != 1 {
		return forest{}, fmt.Errorf("ensemble: only single-output forests are supported, got n_outputs=%d", j.NOutputs)
	}
	trees := make([]*tree.Tree, len(j.Trees))
	for i := range j.Trees {
		t, err := j.Trees[i].Build()
		if err != nil {
			return forest{}, fmt.Errorf("tree %d: %w", i, err)
		}
		trees[i] = t
	}
	var opts []Option
	if len(j.FeatureNames) > 0 {
		opts = append(opts, WithFeatureNames(j.FeatureNames))
	}
	return newForest(j.NFeatures, []float64(j.Classes), trees, opts)
}

// LoadForest decodes whichever tree-ensemble classifier a go-ml/v1 export
// envelope holds, applying the given options. Use it when the estimator type is
// a property of the model file rather than of the code: it serves
// RandomForestClassifier and ExtraTreesClassifier exports alike. The concrete
// loaders ([LoadRandomForestClassifier], [LoadExtraTreesClassifier]) are the
// choice when the type is known and fixed.
func LoadForest(data []byte, opts ...Option) (Forest, error) {
	m, err := goml.LoadBytes(data)
	if err != nil {
		return nil, err
	}
	f, ok := m.(Forest)
	if !ok {
		return nil, fmt.Errorf("ensemble: export is a %s, not a tree-ensemble classifier", m.Type())
	}
	o := applyOptions(opts)
	if o.workers == 0 && o.featureNames == nil {
		return f, nil
	}
	// Every Forest this package decodes embeds *forest and so is tunable; a
	// Forest implemented elsewhere need not be.
	tunable, ok := f.(interface {
		setWorkers(int)
		setFeatureNames([]string) error
	})
	if !ok {
		return nil, fmt.Errorf("ensemble: %s does not support these options", f.Type())
	}
	if o.workers != 0 {
		tunable.setWorkers(o.workers)
	}
	if o.featureNames != nil {
		if err := tunable.setFeatureNames(o.featureNames); err != nil {
			return nil, err
		}
	}
	return f, nil
}
