package goml_test

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	goml "github.com/heru-opensource/go-ml"
	_ "github.com/heru-opensource/go-ml/ensemble" // registers the tree-ensemble types
)

func TestRegisteredTypes(t *testing.T) {
	types := goml.RegisteredTypes()
	for _, want := range []string{"RandomForestClassifier", "ExtraTreesClassifier"} {
		if !slices.Contains(types, want) {
			t.Errorf("%s not registered; have %v", want, types)
		}
	}
}

func TestLoadErrors(t *testing.T) {
	tests := map[string]struct {
		blob string
		is   error
	}{
		"bad format":   {`{"format":"go-ml/v999","type":"X","model":{}}`, goml.ErrFormat},
		"unknown type": {`{"format":"go-ml/v1","type":"NoSuchModel","model":{}}`, goml.ErrUnknownType},
		"not json":     {`{not json`, nil},
	}
	for name, tc := range tests {
		_, err := goml.LoadBytes([]byte(tc.blob))
		if err == nil {
			t.Errorf("%s: expected error", name)
			continue
		}
		if tc.is != nil && !errors.Is(err, tc.is) {
			t.Errorf("%s: err = %v, want errors.Is %v", name, err, tc.is)
		}
	}
}

func TestLoadClassifierNotClassifier(t *testing.T) {
	// Register a model type that is not a Classifier and confirm the classifier
	// loader rejects it with ErrNotClassifier.
	goml.Register("FakeRegressor", func(json.RawMessage) (goml.Model, error) {
		return fakeRegressor{}, nil
	})
	blob := `{"format":"go-ml/v1","type":"FakeRegressor","model":{}}`
	if _, err := goml.LoadClassifierBytes([]byte(blob)); !errors.Is(err, goml.ErrNotClassifier) {
		t.Errorf("err = %v, want ErrNotClassifier", err)
	}
}

type fakeRegressor struct{}

func (fakeRegressor) Type() string                             { return "FakeRegressor" }
func (fakeRegressor) NFeatures() int                           { return 1 }
func (fakeRegressor) Predict(X [][]float64) ([]float64, error) { return make([]float64, len(X)), nil }

func TestRegisterPanics(t *testing.T) {
	assertPanic(t, "nil decoder", func() { goml.Register("WithNilDecoder", nil) })
	goml.Register("DupType", func(json.RawMessage) (goml.Model, error) { return nil, nil })
	assertPanic(t, "duplicate", func() {
		goml.Register("DupType", func(json.RawMessage) (goml.Model, error) { return nil, nil })
	})
}

func TestUnknownTypeMessageIsHelpful(t *testing.T) {
	_, err := goml.LoadBytes([]byte(`{"format":"go-ml/v1","type":"Ghost","model":{}}`))
	if err == nil || !strings.Contains(err.Error(), "did you import") {
		t.Errorf("unknown-type error should hint at importing the package; got %v", err)
	}
}

func assertPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	fn()
}
