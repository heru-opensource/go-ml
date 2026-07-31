// Package ensemble implements tree-ensemble models: static, parallel
// reimplementations of the prediction path of scikit-learn's forests.
//
// Two estimator types ship today:
//
//   - [RandomForestClassifier] — sklearn.ensemble.RandomForestClassifier
//   - [ExtraTreesClassifier] — sklearn.ensemble.ExtraTreesClassifier
//
// They share one implementation because scikit-learn's prediction paths for
// them are identical: every tree yields the class distribution of the leaf a
// sample reaches, and the forest's probability is the mean of those
// distributions. The two estimators differ only in how their trees were grown,
// which is baked into the export. Both satisfy [Forest], and through it
// goml.Classifier.
//
// Importing this package registers its model types with the parent goml
// package, so they can be loaded through goml.Load:
//
//	import _ "github.com/heru-opensource/go-ml/ensemble"
package ensemble
