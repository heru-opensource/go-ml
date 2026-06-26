// Command go-ml-bench measures prediction performance of a go-ml model, as a
// compiled binary, so the numbers are directly comparable to a scikit-learn
// timing of the same model (see tools/sklexport/bench_compare.py).
//
// It reports single-sample latency (the dominant cost in online, one-at-a-time
// serving) and batch throughput, both with goroutine parallelism on and forced
// off:
//
//	go build -o go-ml-bench ./cmd/go-ml-bench
//	./go-ml-bench -model testdata/models/forest_bench.json
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"

	goml "github.com/heru-public/go-ml"
	"github.com/heru-public/go-ml/ensemble"
)

func main() {
	model := flag.String("model", "testdata/models/forest_bench.json", "go-ml/v1 model file")
	dur := flag.Duration("dur", 2*time.Second, "measurement time per case")
	flag.Parse()

	loadStart := time.Now()
	clf, err := goml.LoadClassifierFile(*model)
	loadTime := time.Since(loadStart)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	nf := clf.NFeatures()
	ntrees := "?"
	if rf, ok := clf.(*ensemble.RandomForestClassifier); ok {
		ntrees = fmt.Sprintf("%d", rf.NTrees())
	}
	fmt.Printf("model: %s (%s, %s trees, %d features)\n", *model, clf.Type(), ntrees, nf)
	fmt.Printf("cpus:  %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("load:  %s (parse + build from JSON)\n\n", fmtDur(loadTime))

	// Reuse one model with parallelism and a copy forced sequential.
	data, _ := os.ReadFile(*model)
	seqRF, _ := ensemble.LoadRandomForestClassifier(data, ensemble.WithWorkers(1))

	fmt.Printf("%-26s %12s %14s\n", "case", "per-op", "per-row")
	fmt.Printf("%-26s %12s %14s\n", "----", "------", "-------")

	// Single sample (latency).
	x := randRows(1, nf)
	benchCase("1 row  (parallel)", *dur, 1, func() { sink(clf.PredictProba(x)) })
	benchCase("1 row  (sequential)", *dur, 1, func() { sink(seqRF.PredictProba(x)) })

	// Batches (throughput).
	for _, n := range []int{256, 1000} {
		X := randRows(n, nf)
		benchCase(fmt.Sprintf("%d rows (parallel)", n), *dur, n, func() { sink(clf.PredictProba(X)) })
		benchCase(fmt.Sprintf("%d rows (sequential)", n), *dur, n, func() { sink(seqRF.PredictProba(X)) })
	}
}

func benchCase(name string, dur time.Duration, rows int, fn func()) {
	fn() // warm up
	start := time.Now()
	var iters int64
	for time.Since(start) < dur {
		fn()
		iters++
	}
	elapsed := time.Since(start)
	perOp := elapsed / time.Duration(iters)
	perRow := perOp / time.Duration(rows)
	fmt.Printf("%-26s %12s %14s\n", name, fmtDur(perOp), fmtDur(perRow))
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%d ns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.2f µs", float64(d.Nanoseconds())/1e3)
	default:
		return fmt.Sprintf("%.3f ms", float64(d.Nanoseconds())/1e6)
	}
}

func randRows(n, nf int) [][]float64 {
	rng := rand.New(rand.NewSource(1))
	X := make([][]float64, n)
	for i := range X {
		X[i] = make([]float64, nf)
		for j := range X[i] {
			X[i][j] = rng.NormFloat64() * 500
		}
	}
	return X
}

// sink consumes a result so the compiler cannot eliminate the work.
var checksum float64

func sink(proba [][]float64, err error) {
	if err != nil {
		panic(err)
	}
	checksum += proba[0][0]
}
