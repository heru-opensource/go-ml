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
		"wrong feature count": `{"samples": [[5.9, 3.0, 1.8]]}`,
		"not json":            `{"samples": [[`,
	} {
		if rec, _ := predict(t, body); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status = %d, want 400 (body %s)", name, rec.Code, rec.Body)
		}
	}
}
