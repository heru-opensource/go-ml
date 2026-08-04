package goml

// Model is the behavior common to every model: it knows its scikit-learn
// estimator type name and how many input features it expects. Concrete models
// additionally implement a task interface such as [Classifier] or [Regressor].
type Model interface {
	// Type returns the scikit-learn estimator class name, e.g.
	// "RandomForestClassifier". It matches the "type" field of the export.
	Type() string
	// NFeatures returns the number of input features each sample must have.
	NFeatures() int
	// FeatureNames returns the input feature names in the column order the
	// model expects, or nil when the export carries none — scikit-learn only
	// records them (as feature_names_in_) for an estimator fitted on a named
	// frame, so nil is ordinary and not an error. When they are present, build
	// inputs with an [Assembler] rather than by hand: a feature vector assembled
	// in the wrong order is the one mistake this package cannot catch for you,
	// because every value is individually valid. The returned slice is a copy.
	FeatureNames() []string
}

// Classifier predicts class membership, mirroring scikit-learn's classifier
// API. In every method X is a batch of samples (one inner slice per sample,
// each of length [Model.NFeatures]); a missing feature is encoded as math.NaN.
type Classifier interface {
	Model

	// Classes returns the class labels, in the column order used by
	// PredictProba. The returned slice is a copy and may be modified freely.
	Classes() []float64

	// PredictProba returns, for each input sample, a vector of class
	// probabilities aligned with Classes.
	PredictProba(X [][]float64) ([][]float64, error)

	// Predict returns, for each input sample, the label of the most probable
	// class (ties resolved toward the lowest column index, as in scikit-learn).
	Predict(X [][]float64) ([]float64, error)
}

// Regressor predicts continuous targets. It is defined so the generic [Load]
// entry point and registry can serve regression models as they are added; no
// regressor ships in this package yet.
type Regressor interface {
	Model

	// Predict returns one predicted target value per input sample.
	Predict(X [][]float64) ([]float64, error)
}
