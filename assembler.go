package goml

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// ErrNoFeatureNames is returned by [NewAssembler] when the model does not know
// its feature names — scikit-learn records them only for an estimator fitted on
// a named frame. Use errors.Is to test for it.
var ErrNoFeatureNames = errors.New("goml: model carries no feature names")

// ErrUnknownFeature is returned by [Assembler.Row] for a name the model does not
// have. Use errors.Is to test for it.
var ErrUnknownFeature = errors.New("goml: unknown feature")

// An Assembler builds feature vectors in a model's own column order.
//
// It exists because order is part of a model: a vector assembled in the wrong
// order is made entirely of individually valid numbers, so no amount of
// validation downstream can catch it — the model simply predicts confidently
// from nonsense. Passing names instead of positions moves that failure from
// silent to impossible, and it survives a retrain that reorders or renames
// features, because the names travel with the export.
//
// An Assembler is read-only after construction and safe for concurrent use.
type Assembler struct {
	names []string
	index map[string]int
}

// NewAssembler returns an Assembler for m's feature names, or ErrNoFeatureNames
// if the export carried none. It also rejects duplicate names, which would make
// a name ambiguous.
func NewAssembler(m Model) (*Assembler, error) {
	names := m.FeatureNames()
	if len(names) == 0 {
		return nil, fmt.Errorf("%w: %s", ErrNoFeatureNames, m.Type())
	}
	index := make(map[string]int, len(names))
	for i, n := range names {
		if prev, dup := index[n]; dup {
			return nil, fmt.Errorf("goml: duplicate feature name %q at columns %d and %d", n, prev, i)
		}
		index[n] = i
	}
	return &Assembler{names: names, index: index}, nil
}

// Names returns a copy of the feature names, in column order.
func (a *Assembler) Names() []string { return append([]string(nil), a.names...) }

// Row builds one feature vector from named values.
//
// A name the model does not have is an error wrapping [ErrUnknownFeature] — a
// typo or a stale caller is exactly what this type exists to catch. A name the
// model does have but values omits is math.NaN, the package's missing-feature
// convention, which tree models route natively.
func (a *Assembler) Row(values map[string]float64) ([]float64, error) {
	if err := a.checkNames(values); err != nil {
		return nil, err
	}
	row := make([]float64, len(a.names))
	for i, name := range a.names {
		v, ok := values[name]
		if !ok {
			v = math.NaN()
		}
		row[i] = v
	}
	return row, nil
}

// Rows is Row for a batch, and reports which sample failed.
func (a *Assembler) Rows(values []map[string]float64) ([][]float64, error) {
	X := make([][]float64, len(values))
	for i, v := range values {
		row, err := a.Row(v)
		if err != nil {
			return nil, fmt.Errorf("sample %d: %w", i, err)
		}
		X[i] = row
	}
	return X, nil
}

// Missing returns the names the model expects that values does not supply, in
// column order. [Assembler.Row] treats these as missing features; call this
// first when a missing input should be rejected instead, or logged.
func (a *Assembler) Missing(values map[string]float64) []string {
	var missing []string
	for _, name := range a.names {
		if _, ok := values[name]; !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

func (a *Assembler) checkNames(values map[string]float64) error {
	var unknown []string
	for name := range values {
		if _, ok := a.index[name]; !ok {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown) // map iteration order would make the message unstable
	return fmt.Errorf("%w: %s (model has %s)",
		ErrUnknownFeature, strings.Join(unknown, ", "), strings.Join(a.names, ", "))
}
