// Command serve is a runnable example of go-ml in the shape a service uses it:
// the model is embedded in the binary with //go:embed, decoded once at startup,
// and then served concurrently — no Python, no model file, nothing to fetch at
// runtime. Deployment is one static executable.
//
// It starts an HTTP server on a loopback port, sends it a few requests
// (including one with a missing feature, and a batch of concurrent ones), prints
// what came back, and shuts down:
//
//	go run ./examples/serve
//
// The model is the repository's Iris RandomForestClassifier — 4 features, 3
// classes. iris.json is a copy of testdata/models/iris.json, refreshed by
// `make gen`, because //go:embed cannot reach outside its own directory.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"sync"
	"time"

	goml "github.com/heru-opensource/go-ml"
	_ "github.com/heru-opensource/go-ml/ensemble" // registers the tree-ensemble models
)

//go:embed iris.json
var modelJSON []byte

// The model is decoded once, at process start. A loaded model is immutable and
// safe for concurrent use, so every request shares this one value — there is no
// pool, no lock, and no per-request setup.
var clf = mustLoad()

func mustLoad() goml.Classifier {
	c, err := goml.LoadClassifierBytes(modelJSON)
	if err != nil {
		log.Fatalf("serve: decoding the embedded model: %v", err)
	}
	return c
}

// request is the wire format. JSON has no NaN, so a missing feature is null —
// the same convention most callers end up with.
type request struct {
	Samples [][]*float64 `json:"samples"`
}

type response struct {
	Model         string      `json:"model"`
	Classes       []float64   `json:"classes"`
	Labels        []float64   `json:"labels"`
	Probabilities [][]float64 `json:"probabilities"`
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	var req request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	X := make([][]float64, len(req.Samples))
	for i, sample := range req.Samples {
		X[i] = make([]float64, len(sample))
		for j, v := range sample {
			if v == nil {
				X[i][j] = math.NaN() // absent feature, routed natively by the trees
				continue
			}
			X[i][j] = *v
		}
	}

	proba, err := clf.PredictProba(X)
	if err != nil {
		// A wrong feature count is the caller's mistake, so it is a 400 rather
		// than a 500. It is also the only input error the model reports.
		status := http.StatusInternalServerError
		if errors.Is(err, goml.ErrNumFeatures) {
			status = http.StatusBadRequest
		}
		http.Error(w, err.Error(), status)
		return
	}
	labels, err := clf.Predict(X)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response{
		Model:         clf.Type(),
		Classes:       clf.Classes(),
		Labels:        labels,
		Probabilities: proba,
	}); err != nil {
		log.Printf("serve: writing response: %v", err)
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", handlePredict)

	ln, err := net.Listen("tcp", "127.0.0.1:0") // :0 = any free port
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Print(err)
		}
	}()
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	}()

	url := "http://" + ln.Addr().String() + "/predict"
	fmt.Printf("%s embedded (%d KiB of JSON), %d features, classes %v\n",
		clf.Type(), len(modelJSON)/1024, clf.NFeatures(), clf.Classes())
	fmt.Printf("listening on %s\n\n", ln.Addr())

	// 1. An ordinary batch: a typical setosa and a typical virginica.
	show(url, `{"samples": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}`)

	// 2. A sample whose third feature was never measured.
	show(url, `{"samples": [[5.9, 3.0, null, 1.8]]}`)

	// 3. A malformed request: three features where the model wants four.
	show(url, `{"samples": [[5.9, 3.0, 1.8]]}`)

	// 4. Concurrent traffic against the one shared model.
	concurrent(url, 8, `{"samples": [[6.7, 3.0, 5.2, 2.3]]}`)
}

// show posts one request and prints the outcome.
func show(url, body string) {
	status, payload, err := post(url, body)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("POST %s\n", body)
	if status != http.StatusOK {
		fmt.Printf("  %d %s\n\n", status, payload)
		return
	}
	var res response
	if err := json.Unmarshal([]byte(payload), &res); err != nil {
		log.Fatal(err)
	}
	for i := range res.Labels {
		fmt.Printf("  class=%g proba=%.3f\n", res.Labels[i], res.Probabilities[i])
	}
	fmt.Println()
}

// concurrent fires n identical requests at once and checks they all agree,
// which is the property that lets one loaded model serve every handler.
func concurrent(url string, n int, body string) {
	var wg sync.WaitGroup
	results := make([]string, n)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, payload, err := post(url, body)
			if err != nil {
				log.Fatal(err)
			}
			results[i] = payload
		}(i)
	}
	wg.Wait()

	agreed := true
	for _, r := range results {
		agreed = agreed && r == results[0]
	}
	fmt.Printf("%d concurrent requests, identical responses: %v\n", n, agreed)
}

func post(url, body string) (int, string, error) {
	resp, err := http.Post(url, "application/json", bytes.NewReader([]byte(body)))
	if err != nil {
		return 0, "", err
	}
	defer func() { _ = resp.Body.Close() }()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, "", err
	}
	return resp.StatusCode, string(bytes.TrimSpace(payload)), nil
}
