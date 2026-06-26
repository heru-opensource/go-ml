package jsonx

import (
	"encoding/json"
	"math"
	"testing"
)

func TestFloatsUnmarshal(t *testing.T) {
	var fs Floats
	if err := json.Unmarshal([]byte(`[1, 2.5, -3, 1e30, "Infinity", "-Infinity", "NaN"]`), &fs); err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2.5, -3, 1e30, math.Inf(1), math.Inf(-1), math.NaN()}
	if len(fs) != len(want) {
		t.Fatalf("len = %d, want %d", len(fs), len(want))
	}
	for i := range want {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(fs[i]) {
				t.Errorf("element %d = %v, want NaN", i, fs[i])
			}
			continue
		}
		if fs[i] != want[i] {
			t.Errorf("element %d = %v, want %v", i, fs[i], want[i])
		}
	}
}

func TestFloatsExactRoundTrip(t *testing.T) {
	// Finite float64 values must decode to identical bits via their shortest
	// decimal representation — the property the export format relies on.
	vals := []float64{
		0.1, 1.0 / 3.0, 1437.75, 0.3358080387115479,
		math.MaxFloat64, math.SmallestNonzeroFloat64, -2.0,
	}
	b, _ := json.Marshal(vals) // standard marshal of finite values
	var fs Floats
	if err := json.Unmarshal(b, &fs); err != nil {
		t.Fatal(err)
	}
	for i, v := range vals {
		if math.Float64bits(fs[i]) != math.Float64bits(v) {
			t.Errorf("value %d round-tripped to different bits: got %v want %v", i, fs[i], v)
		}
	}
}

func TestFloatsNullAndErrors(t *testing.T) {
	var fs Floats
	if err := json.Unmarshal([]byte(`null`), &fs); err != nil || fs != nil {
		t.Errorf("null: err=%v fs=%v, want nil,nil", err, fs)
	}
	for _, bad := range []string{`42`, `{"a":1}`, `[true]`, `["bogus"]`, `[1,]`} {
		var x Floats
		if err := json.Unmarshal([]byte(bad), &x); err == nil {
			t.Errorf("expected error decoding %q, got none", bad)
		}
	}
}
