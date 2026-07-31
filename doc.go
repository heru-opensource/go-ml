// Package goml runs trained machine-learning models in pure Go, with output
// that matches the original scikit-learn estimator.
//
// # Goals
//
// The package is built around three ideas:
//
//   - Static, not dynamic. A model is exported once from Python (see
//     tools/sklexport) into a portable go-ml/v1 JSON document. That document is
//     either loaded at startup or, for maximum speed and zero runtime
//     dependencies, compiled directly into your binary as Go source with the
//     go-ml-gen tool (see cmd/go-ml-gen). There is no Python, no pickle, and no
//     model file at runtime.
//   - Generic. Models are values behind small interfaces ([Model], [Classifier],
//     [Regressor]). New estimator types register a decoder with [Register] and
//     are then usable through the same [Load] entry point, mirroring how
//     image.RegisterFormat or database/sql drivers work. The models that ship
//     today are RandomForestClassifier and ExtraTreesClassifier, both in
//     package github.com/heru-public/go-ml/ensemble.
//   - Faithful. Each model reproduces scikit-learn's prediction path exactly,
//     down to its internal float32 cast and missing-value handling, so Go and
//     Python agree to within floating-point rounding.
//
// # Loading a model
//
// Import the package that implements your model type for its side-effect
// registration, then load by type from the export envelope:
//
//	import (
//		goml "github.com/heru-public/go-ml"
//		_ "github.com/heru-public/go-ml/ensemble" // registers the forest models
//	)
//
//	clf, err := goml.LoadClassifierFile("model.json")
//	if err != nil { ... }
//	proba, err := clf.PredictProba([][]float64{{1, 2, 3, ...}})
//
// The API mirrors scikit-learn: [Classifier.PredictProba] returns per-class
// probabilities in [Classifier.Classes] order, and [Classifier.Predict] returns
// the arg-max class label.
//
// # Missing features
//
// Tree models handle missing values natively. Pass math.NaN for an absent
// feature; it is routed exactly as scikit-learn would route it.
package goml
