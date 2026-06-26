// Package jsonx holds small JSON helpers shared across go-ml packages.
//
// Its reason for existing is non-finite floating-point values. Standard JSON
// has no way to spell ±Infinity or NaN, and Go's encoding/json rejects the
// bare Infinity/NaN tokens that Python's json module emits. Decision-tree
// split thresholds, however, are legitimately ±Inf (scikit-learn uses an
// infinite threshold for pure missing-value splits), so the go-ml export
// format encodes non-finite values as the JSON strings "Infinity",
// "-Infinity" and "NaN". [Floats] decodes arrays containing either numbers or
// those sentinels; finite values round-trip exactly.
package jsonx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
)

// Floats is a []float64 that unmarshals from a JSON array whose elements are
// either JSON numbers or the string sentinels "Infinity", "-Infinity" and
// "NaN" (case-insensitive variants and Go's "Inf"/"NaN" spellings are also
// accepted). A JSON null decodes to a nil slice. Finite numbers are parsed
// with correct rounding, so they reproduce the original float64 bit-for-bit.
type Floats []float64

// UnmarshalJSON implements [encoding/json.Unmarshaler].
func (fs *Floats) UnmarshalJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()

	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok == nil { // JSON null
		*fs = nil
		return nil
	}
	if d, ok := tok.(json.Delim); !ok || d != '[' {
		return fmt.Errorf("jsonx: expected JSON array of floats, got %v", tok)
	}

	out := (*fs)[:0]
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		f, err := tokenToFloat(tok)
		if err != nil {
			return err
		}
		out = append(out, f)
	}
	if _, err := dec.Token(); err != nil { // consume closing ']'
		return err
	}
	*fs = out
	return nil
}

func tokenToFloat(tok json.Token) (float64, error) {
	switch v := tok.(type) {
	case json.Number:
		return strconv.ParseFloat(v.String(), 64)
	case float64:
		return v, nil
	case string:
		return parseSentinel(v)
	default:
		return 0, fmt.Errorf("jsonx: invalid float token %v (%T)", tok, tok)
	}
}

func parseSentinel(s string) (float64, error) {
	switch s {
	case "Infinity", "+Infinity", "Inf", "+Inf", "inf":
		return math.Inf(1), nil
	case "-Infinity", "-Inf", "-inf":
		return math.Inf(-1), nil
	case "NaN", "nan", "NAN":
		return math.NaN(), nil
	}
	// Fall back to ParseFloat, which also understands "Inf"/"NaN".
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f, nil
	}
	return 0, fmt.Errorf("jsonx: invalid float string %q", s)
}
