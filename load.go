package goml

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
)

// Format is the export envelope version this package reads.
const Format = "go-ml/v1"

// Errors returned by the loading functions. Use errors.Is to test for them.
var (
	// ErrFormat is returned when the export envelope has an unsupported format.
	ErrFormat = errors.New("goml: unsupported export format")
	// ErrUnknownType is returned by Load when no decoder is registered for the
	// model's type. Importing the package that implements the model (for its
	// side-effect Register call) fixes this.
	ErrUnknownType = errors.New("goml: unknown model type")
	// ErrNotClassifier is returned by the classifier loaders when the loaded
	// model does not implement Classifier.
	ErrNotClassifier = errors.New("goml: model is not a Classifier")
	// ErrNumFeatures is returned by a model's predict methods when an input
	// sample has the wrong number of features.
	ErrNumFeatures = errors.New("goml: wrong number of features")
)

// A Decoder builds a Model from the type-specific "model" object of an export
// envelope. Model implementations register one with [Register].
type Decoder func(model json.RawMessage) (Model, error)

var (
	registryMu sync.RWMutex
	registry   = map[string]Decoder{}
)

// Register installs a decoder for a model type (the value of the envelope's
// "type" field, e.g. "RandomForestClassifier"). It is normally called from a
// model package's init function. Register panics if dec is nil or if typeName
// is already registered, mirroring image.RegisterFormat and sql.Register.
func Register(typeName string, dec Decoder) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if dec == nil {
		panic("goml: Register decoder is nil")
	}
	if _, dup := registry[typeName]; dup {
		panic("goml: Register called twice for type " + typeName)
	}
	registry[typeName] = dec
}

// RegisteredTypes returns the sorted list of model types that have a decoder.
func RegisteredTypes() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	types := make([]string, 0, len(registry))
	for t := range registry {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}

type envelope struct {
	Format string          `json:"format"`
	Type   string          `json:"type"`
	Model  json.RawMessage `json:"model"`
}

// Load reads a go-ml/v1 export from r and constructs the model, dispatching on
// the envelope's "type" to the decoder registered for it.
func Load(r io.Reader) (Model, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return LoadBytes(data)
}

// LoadBytes is Load on an in-memory export. It is the basis of statically
// embedded models: pair it with //go:embed to link the export into the binary.
func LoadBytes(data []byte) (Model, error) {
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("goml: parsing export envelope: %w", err)
	}
	if env.Format != Format {
		if env.Format == BundleFormat {
			return nil, fmt.Errorf("%w: %q holds several models — use LoadBundle", ErrFormat, env.Format)
		}
		return nil, fmt.Errorf("%w: %q (this build understands %q)", ErrFormat, env.Format, Format)
	}
	return decodeModel(env.Type, env.Model)
}

// decodeModel dispatches one type-specific model object to its registered
// decoder. Both the single-model envelope and each entry of a bundle land here.
func decodeModel(typeName string, raw json.RawMessage) (Model, error) {
	registryMu.RLock()
	dec, ok := registry[typeName]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %q (did you import the package that registers it? known: %v)",
			ErrUnknownType, typeName, RegisteredTypes())
	}
	m, err := dec(raw)
	if err != nil {
		return nil, fmt.Errorf("goml: decoding %s: %w", typeName, err)
	}
	return m, nil
}

// LoadFile loads a model from a file containing a go-ml/v1 export.
func LoadFile(path string) (Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := LoadBytes(data)
	if err != nil {
		return nil, fmt.Errorf("goml: %s: %w", path, err)
	}
	return m, nil
}

// LoadClassifier is Load followed by an assertion that the model classifies.
func LoadClassifier(r io.Reader) (Classifier, error) {
	m, err := Load(r)
	if err != nil {
		return nil, err
	}
	return asClassifier(m)
}

// LoadClassifierBytes is LoadBytes followed by a Classifier assertion.
func LoadClassifierBytes(data []byte) (Classifier, error) {
	m, err := LoadBytes(data)
	if err != nil {
		return nil, err
	}
	return asClassifier(m)
}

// LoadClassifierFile is LoadFile followed by a Classifier assertion.
func LoadClassifierFile(path string) (Classifier, error) {
	m, err := LoadFile(path)
	if err != nil {
		return nil, err
	}
	return asClassifier(m)
}

func asClassifier(m Model) (Classifier, error) {
	c, ok := m.(Classifier)
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNotClassifier, m.Type())
	}
	return c, nil
}
