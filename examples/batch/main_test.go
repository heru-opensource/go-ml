package main

import (
	"math"
	"strings"
	"testing"
)

func TestReadCSVSample(t *testing.T) {
	header, X, err := readCSV(strings.NewReader(sampleCSV), 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(header) != 4 || header[0] != "sepal_length" {
		t.Fatalf("header = %v", header)
	}
	if len(X) != 8 {
		t.Fatalf("got %d rows, want 8", len(X))
	}
	// Row 4 of the sample has an empty petal_length: the missing-value case the
	// example exists to show.
	if !math.IsNaN(X[3][2]) {
		t.Errorf("X[3][2] = %v, want NaN", X[3][2])
	}
	for i, row := range X {
		if len(row) != 4 {
			t.Errorf("row %d has %d features, want 4", i, len(row))
		}
	}
}

func TestReadCSVErrors(t *testing.T) {
	cases := map[string]string{
		"empty input":     "",
		"header only":     "a,b,c,d\n",
		"non-numeric":     "a,b,c,d\n1,2,three,4\n",
		"too few columns": "a,b,c,d\n1,2,3\n",
	}
	for name, in := range cases {
		if _, _, err := readCSV(strings.NewReader(in), 4); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

// TestAlignReordersByName is the point of carrying feature names: a CSV whose
// columns are in a different order than the model's must still score correctly.
func TestAlignReordersByName(t *testing.T) {
	names := []string{"a", "b", "c"}
	header := []string{"c", "a", "b"}
	X := [][]float64{{3, 1, 2}, {30, 10, 20}}

	got, err := align(names, header, X)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{{1, 2, 3}, {10, 20, 30}}
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("row %d = %v, want %v", i, got[i], want[i])
			}
		}
	}
}

func TestAlignRejectsAMissingColumn(t *testing.T) {
	_, err := align([]string{"a", "b"}, []string{"a", "bb"}, [][]float64{{1, 2}})
	if err == nil || !strings.Contains(err.Error(), "b") {
		t.Errorf("err = %v, want it to name the missing column", err)
	}
}

// TestAlignWithoutNamesTrustsPosition covers the older exports: with no names
// there is nothing to match on, so the rows pass through untouched.
func TestAlignWithoutNamesTrustsPosition(t *testing.T) {
	X := [][]float64{{1, 2}}
	got, err := align(nil, []string{"x", "y"}, X)
	if err != nil {
		t.Fatal(err)
	}
	if got[0][0] != 1 || got[0][1] != 2 {
		t.Errorf("rows = %v, want them unchanged", got)
	}
}

func TestWriteCSVRoundTripsMissingValues(t *testing.T) {
	var out strings.Builder
	err := writeCSV(&out,
		[]string{"a", "b"},
		[]float64{0, 1},
		[][]float64{{1.5, math.NaN()}},
		[]float64{1},
		[][]float64{{0.25, 0.75}},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "a,b,predicted,p(0),p(1)\n1.5,,1,0.250,0.750\n"
	if out.String() != want {
		t.Errorf("got:\n%q\nwant:\n%q", out.String(), want)
	}
}
