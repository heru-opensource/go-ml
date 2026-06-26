package ensemble_test

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/heru-public/go-ml/ensemble"
)

// loadBenchModel loads the forest_bench model (200 trees, 30 features) exported
// to the repo-root testdata, two directories up from this package.
func loadBenchModel(b *testing.B, workers int) *ensemble.RandomForestClassifier {
	b.Helper()
	path := filepath.Join("..", "testdata", "models", "forest_bench.json")
	data, err := os.ReadFile(path)
	if err != nil {
		b.Skipf("model not available (%v); run `make regen` to generate it", err)
	}
	rf, err := ensemble.LoadRandomForestClassifier(data, ensemble.WithWorkers(workers))
	if err != nil {
		b.Fatal(err)
	}
	return rf
}

func randInputs(n, nf int) [][]float64 {
	rng := rand.New(rand.NewSource(1))
	X := make([][]float64, n)
	for i := range X {
		X[i] = make([]float64, nf)
		for j := range X[i] {
			X[i][j] = rng.NormFloat64() * 3
		}
	}
	return X
}

// BenchmarkPredictProba covers single-row (latency) and batch (throughput)
// prediction, sequential vs. parallel, on the real forest.
func BenchmarkPredictProba(b *testing.B) {
	for _, n := range []int{1, 256, 1000} {
		for _, workers := range []int{1, 0} { // 0 => GOMAXPROCS
			name := "rows=" + itoa(n)
			if workers == 1 {
				name += "/sequential"
			} else {
				name += "/parallel"
			}
			b.Run(name, func(b *testing.B) {
				rf := loadBenchModel(b, workers)
				X := randInputs(n, rf.NFeatures())
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := rf.PredictProba(X); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
