// Package goml runs trained machine-learning models in pure Go, with output
// that matches the original scikit-learn estimator.
//
// # Caution: this library is fully AI-generated
//
// The implementation, tests, examples and documentation were all written by an
// AI agent.
//
// Every shipped model is checked against scikit-learn's own outputs by the test
// suite in the repository, but the only real-world validation is against
// production workloads specific to Heru, Inc. Anything those workloads do not
// exercise — other estimator types, other hyper-parameters, other scikit-learn
// versions, other input regimes — is unproven, and exactness against a moving
// upstream is exactly where that gap is most likely to hurt. Use it with
// caution: read the code, validate your own exported model against your own
// scikit-learn outputs, and treat the fidelity claims below as claims to verify
// rather than as guarantees.
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
//     package github.com/heru-opensource/go-ml/ensemble.
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
//		goml "github.com/heru-opensource/go-ml"
//		_ "github.com/heru-opensource/go-ml/ensemble" // registers the forest models
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
//
// # Bundles
//
// A deployed model is often not one estimator: a decision may take two or three
// of them plus the thresholds they were tuned against, and those thresholds are
// as much a fitted parameter as any split in a tree. [LoadBundle] reads a
// go-ml/bundle-v1 document holding all of it, [Bundle.Classifier] returns a
// member by name, and the metadata accessors return the tuned values typed:
//
//	b, err := goml.LoadBundleFile("cascade.json")
//	screen, err := b.Classifier("screen")
//	threshold, err := b.Float("screen_confidence") // missing key: an error, not a zero
//
// go-ml carries metadata and hands it back; it does not interpret it.
//
// # Feature names
//
// The order of a feature vector is part of a model, and a vector assembled in
// the wrong order is made of individually valid numbers — no validation can
// catch it. When scikit-learn recorded feature_names_in_ (which it does for an
// estimator fitted on a named frame), the export carries those names,
// [Model.FeatureNames] returns them in column order, and [Assembler] builds
// inputs from a name-keyed map:
//
//	a, err := goml.NewAssembler(clf)   // ErrNoFeatureNames if the export has none
//	row, err := a.Row(map[string]float64{"petal_length": 1.4, "sepal_width": 3.5})
//
// An unknown name is an error; a name the model expects but the map omits is a
// missing feature. A retrain that reorders or renames features then changes the
// export rather than silently breaking its callers.
package goml
