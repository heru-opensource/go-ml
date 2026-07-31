// Command batch is a runnable example of offline scoring: read a CSV, hand the
// whole thing to the model in one call, write a CSV back. It is the other half
// of the deployment story from examples/serve — there the model is embedded and
// serves one request at a time; here it is loaded from a file at startup and the
// work arrives in bulk.
//
// Two things worth copying from it: an empty CSV field becomes math.NaN, which
// the trees route natively instead of erroring or imputing; and the whole batch
// goes through a single PredictProba call, which is what lets go-ml spread the
// rows across goroutines.
//
// Run it from the repository root:
//
//	go run ./examples/batch                                     # embedded sample rows
//	go run ./examples/batch -csv rows.csv -workers 4            # your own file
//	go run ./examples/batch -model testdata/models/iris.json    # any go-ml export
package main

import (
	_ "embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/heru-opensource/go-ml/ensemble"
)

//go:embed sample.csv
var sampleCSV string

func main() {
	model := flag.String("model", "testdata/models/iris.json", "go-ml/v1 export to score with")
	input := flag.String("csv", "", "CSV file to score (default: the embedded sample rows)")
	workers := flag.Int("workers", 0, "goroutines per prediction call (0 = GOMAXPROCS, 1 = sequential)")
	flag.Parse()

	// Loading through ensemble.LoadForest rather than goml.LoadClassifierFile
	// buys two things: the worker knob, and NTrees for the report below. It
	// accepts either forest type, so -model can point at any of them.
	data, err := os.ReadFile(*model)
	if err != nil {
		log.Fatalf("batch: %v", err)
	}
	clf, err := ensemble.LoadForest(data, ensemble.WithWorkers(*workers))
	if err != nil {
		log.Fatalf("batch: %v", err)
	}

	header, X, err := readCSV(openInput(*input), clf.NFeatures())
	if err != nil {
		log.Fatalf("batch: %v", err)
	}

	// One call for the whole file. Row-by-row would give identical numbers and
	// waste the batching.
	start := time.Now()
	proba, err := clf.PredictProba(X)
	if err != nil {
		log.Fatalf("batch: %v", err)
	}
	labels, err := clf.Predict(X)
	if err != nil {
		log.Fatalf("batch: %v", err)
	}
	elapsed := time.Since(start)

	if err := writeCSV(os.Stdout, header, clf.Classes(), X, labels, proba); err != nil {
		log.Fatalf("batch: %v", err)
	}

	// Progress and timing go to stderr so the CSV on stdout stays pipeable.
	fmt.Fprintf(os.Stderr, "\n%s: %d trees, %d features — scored %d rows in %s (%.0f rows/s)\n",
		clf.Type(), clf.NTrees(), clf.NFeatures(), len(X), elapsed.Round(time.Microsecond),
		float64(len(X))/elapsed.Seconds())
}

func openInput(path string) io.Reader {
	if path == "" {
		return strings.NewReader(sampleCSV)
	}
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("batch: %v", err)
	}
	// The process exits right after writing, so the file is closed by exit.
	return f
}

// readCSV parses a header row plus one row of features per sample. An empty
// field is a missing feature: math.NaN, which the model routes natively.
func readCSV(r io.Reader, nFeatures int) ([]string, [][]float64, error) {
	rd := csv.NewReader(r)
	rd.FieldsPerRecord = nFeatures

	header, err := rd.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("reading the header: %w", err)
	}

	var X [][]float64
	for line := 2; ; line++ {
		rec, err := rd.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		row := make([]float64, len(rec))
		for i, field := range rec {
			field = strings.TrimSpace(field)
			if field == "" {
				row[i] = math.NaN()
				continue
			}
			v, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, nil, fmt.Errorf("line %d, column %q: %w", line, header[i], err)
			}
			row[i] = v
		}
		X = append(X, row)
	}
	if len(X) == 0 {
		return nil, nil, fmt.Errorf("no data rows")
	}
	return header, X, nil
}

// writeCSV echoes the input columns and appends the prediction: the label plus
// one probability column per class, named after the class label.
func writeCSV(w io.Writer, header []string, classes []float64, X [][]float64, labels []float64, proba [][]float64) error {
	out := csv.NewWriter(w)

	row := append([]string(nil), header...)
	row = append(row, "predicted")
	for _, c := range classes {
		row = append(row, fmt.Sprintf("p(%g)", c))
	}
	if err := out.Write(row); err != nil {
		return err
	}

	for i := range X {
		row = row[:0]
		for _, v := range X[i] {
			if math.IsNaN(v) {
				row = append(row, "") // round-trips as the missing value it was
				continue
			}
			row = append(row, strconv.FormatFloat(v, 'g', -1, 64))
		}
		row = append(row, strconv.FormatFloat(labels[i], 'g', -1, 64))
		for _, p := range proba[i] {
			row = append(row, strconv.FormatFloat(p, 'f', 3, 64))
		}
		if err := out.Write(row); err != nil {
			return err
		}
	}

	out.Flush()
	return out.Error()
}
