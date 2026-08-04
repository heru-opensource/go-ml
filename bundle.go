package goml

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/heru-opensource/go-ml/internal/jsonx"
)

// BundleFormat is the envelope version [LoadBundle] reads.
const BundleFormat = "go-ml/bundle-v1"

// Errors returned by the bundle loaders and accessors. Use errors.Is to test.
var (
	// ErrUnknownModel is returned by [Bundle.Model] and [Bundle.Classifier] for
	// a name the bundle does not contain.
	ErrUnknownModel = errors.New("goml: bundle has no such model")
	// ErrUnknownMeta is returned by the [Bundle] metadata accessors for a key
	// the bundle does not carry.
	ErrUnknownMeta = errors.New("goml: bundle has no such metadata key")
)

// A Bundle is several named models shipped as one artifact, together with the
// scalars that were tuned alongside them.
//
// It exists because a deployed model is often not one estimator. A decision may
// need two or three of them plus the thresholds they were tuned against — and
// those thresholds are as much a fitted parameter as any split in a tree.
// Keeping them in hand-written Go beside the model is what goes stale: the
// numbers and the trees are updated by different hands, at different times, and
// nothing fails until the predictions are quietly wrong. A bundle makes them one
// file, versioned and deployed together.
//
// What a bundle deliberately does not do is interpret its metadata. go-ml
// carries the numbers and hands them back typed; what a threshold means is the
// caller's own logic, and a missing key is a loud error rather than a zero.
//
// A Bundle is read-only after loading and safe for concurrent use.
type Bundle struct {
	models map[string]Model
	meta   map[string]json.RawMessage
}

// NewBundle assembles a bundle from already-built models and raw JSON metadata
// values. This is the constructor statically generated code calls (see
// cmd/go-ml-gen); most callers load a bundle instead.
func NewBundle(models map[string]Model, metadata map[string]json.RawMessage) (*Bundle, error) {
	if len(models) == 0 {
		return nil, errors.New("goml: bundle has no models")
	}
	b := &Bundle{
		models: make(map[string]Model, len(models)),
		meta:   make(map[string]json.RawMessage, len(metadata)),
	}
	for name, m := range models {
		if m == nil {
			return nil, fmt.Errorf("goml: bundle model %q is nil", name)
		}
		b.models[name] = m
	}
	for key, raw := range metadata {
		if !json.Valid(raw) {
			return nil, fmt.Errorf("goml: bundle metadata %q is not valid JSON", key)
		}
		b.meta[key] = raw
	}
	return b, nil
}

// Names returns the model names, sorted.
func (b *Bundle) Names() []string { return sortedKeys(b.models) }

// Model returns the named model, or an error wrapping [ErrUnknownModel] listing
// what the bundle does hold. Type-assert the result for a concrete model's own
// methods.
func (b *Bundle) Model(name string) (Model, error) {
	m, ok := b.models[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q (bundle has %v)", ErrUnknownModel, name, b.Names())
	}
	return m, nil
}

// Classifier is [Bundle.Model] followed by a [Classifier] assertion.
func (b *Bundle) Classifier(name string) (Classifier, error) {
	m, err := b.Model(name)
	if err != nil {
		return nil, err
	}
	return asClassifier(m)
}

// MetaKeys returns the metadata keys, sorted.
func (b *Bundle) MetaKeys() []string { return sortedKeys(b.meta) }

// Float returns a numeric metadata value. The non-finite sentinels the export
// format uses ("Infinity", "NaN") decode here too.
func (b *Bundle) Float(key string) (float64, error) {
	raw, err := b.raw(key)
	if err != nil {
		return 0, err
	}
	f, err := jsonx.Float(raw)
	if err != nil {
		return 0, fmt.Errorf("goml: bundle metadata %q: %w", key, err)
	}
	return f, nil
}

// Int returns an integer metadata value, rejecting a number with a fractional
// part rather than truncating it.
func (b *Bundle) Int(key string) (int, error) {
	f, err := b.Float(key)
	if err != nil {
		return 0, err
	}
	n := int(f)
	if float64(n) != f {
		return 0, fmt.Errorf("goml: bundle metadata %q is %v, not an integer", key, f)
	}
	return n, nil
}

// String returns a string metadata value.
func (b *Bundle) String(key string) (string, error) {
	var s string
	if err := b.Meta(key, &s); err != nil {
		return "", err
	}
	return s, nil
}

// Bool returns a boolean metadata value.
func (b *Bundle) Bool(key string) (bool, error) {
	var v bool
	if err := b.Meta(key, &v); err != nil {
		return false, err
	}
	return v, nil
}

// Meta decodes a metadata value into v, for anything the typed accessors do not
// cover — a list of thresholds, say, or a small object.
func (b *Bundle) Meta(key string, v any) error {
	raw, err := b.raw(key)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("goml: bundle metadata %q: %w", key, err)
	}
	return nil
}

func (b *Bundle) raw(key string) (json.RawMessage, error) {
	raw, ok := b.meta[key]
	if !ok {
		return nil, fmt.Errorf("%w: %q (bundle has %v)", ErrUnknownMeta, key, b.MetaKeys())
	}
	return raw, nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// --- loading ---

type bundleEnvelope struct {
	Format   string                     `json:"format"`
	Models   map[string]bundleEntry     `json:"models"`
	Metadata map[string]json.RawMessage `json:"metadata"`
}

type bundleEntry struct {
	Type  string          `json:"type"`
	Model json.RawMessage `json:"model"`
}

// LoadBundle reads a go-ml/bundle-v1 document from r and decodes every model in
// it, dispatching each on its own "type" exactly as [Load] does.
func LoadBundle(r io.Reader) (*Bundle, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return LoadBundleBytes(data)
}

// LoadBundleBytes is [LoadBundle] on an in-memory document, for pairing with
// //go:embed.
func LoadBundleBytes(data []byte) (*Bundle, error) {
	var env bundleEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("goml: parsing bundle envelope: %w", err)
	}
	if env.Format != BundleFormat {
		if env.Format == Format {
			return nil, fmt.Errorf("%w: %q holds a single model — use Load", ErrFormat, env.Format)
		}
		return nil, fmt.Errorf("%w: %q (this build understands %q)", ErrFormat, env.Format, BundleFormat)
	}
	if len(env.Models) == 0 {
		return nil, errors.New("goml: bundle has no models")
	}

	models := make(map[string]Model, len(env.Models))
	for name, entry := range env.Models {
		m, err := decodeModel(entry.Type, entry.Model)
		if err != nil {
			return nil, fmt.Errorf("goml: bundle model %q: %w", name, err)
		}
		models[name] = m
	}
	return NewBundle(models, env.Metadata)
}

// LoadBundleFile loads a bundle from a file.
func LoadBundleFile(path string) (*Bundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	b, err := LoadBundleBytes(data)
	if err != nil {
		return nil, fmt.Errorf("goml: %s: %w", path, err)
	}
	return b, nil
}
