package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func predict(t *testing.T, body string) (*httptest.ResponseRecorder, response) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(body))
	rec := httptest.NewRecorder()
	handlePredict(rec, req)

	var res response
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
			t.Fatalf("decoding response %q: %v", rec.Body.String(), err)
		}
	}
	return rec, res
}

func TestPredictBatch(t *testing.T) {
	rec, res := predict(t, `{"samples": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	if len(res.Labels) != 2 || len(res.Probabilities) != 2 {
		t.Fatalf("got %d labels and %d probability rows, want 2 of each",
			len(res.Labels), len(res.Probabilities))
	}
	// A typical setosa and a typical virginica, the first and last classes.
	if res.Labels[0] != res.Classes[0] || res.Labels[1] != res.Classes[len(res.Classes)-1] {
		t.Errorf("labels = %v, want [%g %g]", res.Labels, res.Classes[0], res.Classes[len(res.Classes)-1])
	}
	for i, row := range res.Probabilities {
		var sum float64
		for _, p := range row {
			sum += p
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("row %d probabilities sum to %v, want 1", i, sum)
		}
	}
}

// TestPredictMissingFeature pins the wire convention: JSON has no NaN, so an
// absent feature is null, and the trees route it natively rather than erroring.
func TestPredictMissingFeature(t *testing.T) {
	rec, res := predict(t, `{"samples": [[5.9, 3.0, null, 1.8]]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	var sum float64
	for _, p := range res.Probabilities[0] {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum to %v, want 1", sum)
	}
}

func TestPredictRejectsBadInput(t *testing.T) {
	for name, body := range map[string]string{
		"wrong feature count":  `{"samples": [[5.9, 3.0, 1.8]]}`,
		"not json":             `{"samples": [[`,
		"unknown feature name": `{"rows": [{"petal_len": 1.4}]}`,
		"unknown among knowns": `{"rows": [{"sepal_length": 5.1, "nope": 1}]}`,
	} {
		if rec, _ := predict(t, body); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400 (body %s)", name, rec.Code, rec.Body)
		}
	}
}

// TestNamedRowMatchesPositional is the reason "rows" exists: the same values
// keyed by name, in a different order, must predict identically to the
// positional form — and cannot be silently misordered.
func TestNamedRowMatchesPositional(t *testing.T) {
	recPos, pos := predict(t, `{"samples": [[5.1, 3.5, 1.4, 0.2]]}`)
	recNamed, named := predict(t, `{"rows": [{"petal_width": 0.2, "sepal_length": 5.1, "petal_length": 1.4, "sepal_width": 3.5}]}`)
	if recPos.Code != http.StatusOK || recNamed.Code != http.StatusOK {
		t.Fatalf("statuses = %d and %d", recPos.Code, recNamed.Code)
	}
	if pos.Labels[0] != named.Labels[0] {
		t.Errorf("labels differ: positional %g, named %g", pos.Labels[0], named.Labels[0])
	}
	for c := range pos.Probabilities[0] {
		if pos.Probabilities[0][c] != named.Probabilities[0][c] {
			t.Errorf("probabilities differ: %v vs %v", pos.Probabilities[0], named.Probabilities[0])
			break
		}
	}
}

// TestNamedRowTreatsOmittedFeatureAsMissing pins the other half of the
// convention: an absent key is a missing feature, not an error.
func TestNamedRowTreatsOmittedFeatureAsMissing(t *testing.T) {
	rec, res := predict(t, `{"rows": [{"sepal_length": 5.9, "sepal_width": 3.0, "petal_width": 1.8}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	var sum float64
	for _, p := range res.Probabilities[0] {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("probabilities sum to %v, want 1", sum)
	}
}
