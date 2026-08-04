// Command go-ml-gen compiles a go-ml/v1 model export into Go source, so the
// model is linked statically into your binary with no runtime file or parsing.
//
// Given an export produced by tools/sklexport, it emits a .go file that builds
// the model from literal data and exposes it as a package-level variable:
//
//	go-ml-gen -pkg models -var Model -o models/model_gen.go model.json
//
// Then in your program:
//
//	proba, err := models.Model.PredictProba(X)
//
// Multiple exports can be generated into the same package; pass several inputs
// and an output directory:
//
//	go-ml-gen -pkg models -o models/ iris.json forest.json
//
// The variable name then defaults to the CamelCase of each file's base name.
// Generated files carry a "DO NOT EDIT" banner and are gofmt-clean.
//
// A bundle export (go-ml/bundle-v1) is compiled in the same way, producing a
// *goml.Bundle with one builder per model and the tuned metadata as literals.
//
// # Models that do not exist yet
//
// Every generated file also defines <Var>Available() bool, reporting true. The
// -stub flag emits a placeholder declaring exactly the same names, with a nil
// var and Available() reporting false:
//
//	go-ml-gen -stub -pkg models -var Model -o models/model_gen.go
//	go-ml-gen -stub -type Bundle -pkg models -var Cascade -o models/cascade_gen.go
//
// A package that consumes a model can then compile before anyone has trained
// one — during bootstrapping, in a fork that does not ship the artifact, in CI
// on a machine without it — without hand-writing a nil var of its own. Guard use
// with Available(), commit the placeholder, and regenerate over it from a real
// export when there is one; no call site changes and no build tags are involved.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/heru-opensource/go-ml/internal/jsonx"
)

func main() {
	pkg := flag.String("pkg", "models", "package name for generated files")
	varName := flag.String("var", "", "exported variable name (single input only; default: CamelCase of file base)")
	out := flag.String("o", "", "output file (single input) or directory (multiple inputs); default: <input>_gen.go")
	stub := flag.Bool("stub", false, "emit a placeholder for a model that is not available yet, "+
		"declaring the same names as real generated code (takes no input; needs -var and -o)")
	stubType := flag.String("type", typeRandomForest, "concrete type for -stub: "+
		typeRandomForest+", "+typeExtraTrees+" or "+typeBundle)
	flag.Parse()

	inputs := flag.Args()

	if *stub {
		if err := generateStub(*out, *pkg, *varName, *stubType, inputs); err != nil {
			fmt.Fprintf(os.Stderr, "go-ml-gen: %v\n", err)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stderr, "go-ml-gen: wrote %s (placeholder var %s.%s, %sAvailable() == false)\n",
			*out, *pkg, *varName, *varName)
		return
	}

	if len(inputs) == 0 {
		fmt.Fprintln(os.Stderr, "go-ml-gen: no input exports given")
		flag.Usage()
		os.Exit(2)
	}
	if *varName != "" && len(inputs) != 1 {
		fmt.Fprintln(os.Stderr, "go-ml-gen: -var may only be used with a single input")
		os.Exit(2)
	}

	for _, in := range inputs {
		name := *varName
		if name == "" {
			name = exportVarName(in)
		}
		outPath := outputPath(*out, in, len(inputs))
		if err := generate(in, outPath, *pkg, name); err != nil {
			fmt.Fprintf(os.Stderr, "go-ml-gen: %s: %v\n", in, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "go-ml-gen: wrote %s (var %s.%s)\n", outPath, *pkg, name)
	}
}

func outputPath(out, in string, n int) string {
	base := strings.TrimSuffix(filepath.Base(in), filepath.Ext(in))
	if out == "" {
		return filepath.Join(filepath.Dir(in), base+"_gen.go")
	}
	if n > 1 || isDir(out) {
		return filepath.Join(out, base+"_gen.go")
	}
	return out
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// --- go-ml/v1 schema (a local copy; the tool deliberately does not import the
// model packages so it can be built and run without them). ---

const (
	formatV1     = "go-ml/v1"
	formatBundle = "go-ml/bundle-v1"

	typeRandomForest = "RandomForestClassifier"
	typeExtraTrees   = "ExtraTreesClassifier"
	typeBundle       = "Bundle"
)

// envelope covers both documents: a single model carries type/model, a bundle
// carries models/metadata. Format says which.
type envelope struct {
	Format   string                     `json:"format"`
	Type     string                     `json:"type"`
	Model    json.RawMessage            `json:"model"`
	Models   map[string]bundleEntry     `json:"models"`
	Metadata map[string]json.RawMessage `json:"metadata"`
}

type bundleEntry struct {
	Type  string          `json:"type"`
	Model json.RawMessage `json:"model"`
}

type treeJSON struct {
	NodeCount   int          `json:"node_count"`
	ValueWidth  int          `json:"value_width"`
	Left        []int32      `json:"left"`
	Right       []int32      `json:"right"`
	Feature     []int32      `json:"feature"`
	Threshold   jsonx.Floats `json:"threshold"`
	MissingLeft []bool       `json:"missing_left"`
	Value       jsonx.Floats `json:"value"`
}

type forestJSON struct {
	NFeatures    int          `json:"n_features"`
	NOutputs     int          `json:"n_outputs"`
	Classes      jsonx.Floats `json:"classes"`
	FeatureNames []string     `json:"feature_names"`
	Trees        []treeJSON   `json:"trees"`
}

func generate(in, outPath, pkg, varName string) error {
	data, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return fmt.Errorf("parsing envelope: %w", err)
	}

	var body bytes.Buffer
	var imports importSet
	var err2 error
	switch env.Format {
	case formatV1:
		err2 = emitSingle(&body, &imports, varName, &env)
	case formatBundle:
		err2 = emitBundle(&body, &imports, varName, &env)
	default:
		return fmt.Errorf("unsupported format %q", env.Format)
	}
	if err2 != nil {
		return err2
	}

	emitAvailable(&body, varName)
	return writeSource(outPath, pkg, "go-ml-gen from "+filepath.Base(in), &imports, &body)
}

// generateStub writes a placeholder for a model that does not exist yet.
//
// A package that consumes a generated model still has to compile before anyone
// has trained one — during bootstrapping, in a fork that does not ship the
// artifact, in CI on a machine without it. The usual answer is a hand-written
// nil var, reinvented per caller. This emits that file instead, declaring
// exactly the names real generated code declares, so nothing at the call site
// changes when the model arrives: regenerate over this file from an export and
// <Var>Available() starts returning true.
func generateStub(outPath, pkg, varName, modelType string, inputs []string) error {
	switch {
	case len(inputs) > 0:
		return fmt.Errorf("-stub takes no input exports (got %v); it exists for when there is no export yet", inputs)
	case varName == "":
		return fmt.Errorf("-stub needs -var to know what to declare")
	case outPath == "":
		return fmt.Errorf("-stub needs -o")
	}

	var imports importSet
	var goType string
	switch modelType {
	case typeRandomForest, typeExtraTrees:
		imports.ensemble = true
		goType = "*ensemble." + modelType
	case typeBundle:
		imports.goml = true
		goType = "*goml.Bundle"
	default:
		return fmt.Errorf("unsupported -type %q (want %s, %s or %s)",
			modelType, typeRandomForest, typeExtraTrees, typeBundle)
	}

	var body bytes.Buffer
	fmt.Fprintf(&body, "// %s is nil: no model export was available when this file was generated.\n//\n", varName)
	fmt.Fprintf(&body, "// Real generated code declares the same names, so call sites compile either\n")
	fmt.Fprintf(&body, "// way — but they must ask %sAvailable() first, because methods on a nil\n", varName)
	fmt.Fprintf(&body, "// model panic. Regenerate over this file from an export to compile one in.\n")
	fmt.Fprintf(&body, "var %s %s\n\n", varName, goType)
	fmt.Fprintf(&body, "// %sAvailable reports whether %s was compiled from a model export.\n", varName, varName)
	fmt.Fprintf(&body, "// It is false here and true in code generated from a real export.\n")
	fmt.Fprintf(&body, "func %sAvailable() bool { return false }\n", varName)

	return writeSource(outPath, pkg, "go-ml-gen (-stub)", &imports, &body)
}

// emitAvailable writes the companion of the stub's <Var>Available: the same
// function, reporting true, so a caller's guard needs no build tags and no edit
// when the model lands.
func emitAvailable(w *bytes.Buffer, varName string) {
	fmt.Fprintf(w, "\n// %sAvailable reports whether %s was compiled from a model export.\n", varName, varName)
	fmt.Fprintf(w, "// It is true here; the -stub placeholder defines the same function returning\n")
	fmt.Fprintf(w, "// false, so a caller can be written once for both.\n")
	fmt.Fprintf(w, "func %sAvailable() bool { return true }\n", varName)
}

// importSet collects what the emitted code turned out to need. Generated files
// import only that: "math" appears solely for non-finite literals, and
// "encoding/json"/goml only for a bundle's metadata.
type importSet struct {
	math     bool // math.Inf / math.NaN literals
	json     bool // json.RawMessage metadata values
	goml     bool // goml.Bundle
	ensemble bool // the model constructors
	tree     bool // tree.Tree literals
}

func (s *importSet) render() string {
	var b strings.Builder
	b.WriteString("import (\n")
	if s.json {
		b.WriteString("\t\"encoding/json\"\n")
	}
	if s.math {
		b.WriteString("\t\"math\"\n")
	}
	if s.json || s.math {
		b.WriteString("\n")
	}
	if s.goml {
		b.WriteString("\tgoml \"github.com/heru-opensource/go-ml\"\n")
	}
	if s.ensemble {
		b.WriteString("\t\"github.com/heru-opensource/go-ml/ensemble\"\n")
	}
	if s.tree {
		b.WriteString("\t\"github.com/heru-opensource/go-ml/tree\"\n")
	}
	b.WriteString(")\n\n")
	return b.String()
}

// writeSource assembles the banner, package clause, imports and body into a
// gofmt-clean file.
func writeSource(outPath, pkg, generatedBy string, imports *importSet, body *bytes.Buffer) error {
	var src bytes.Buffer
	fmt.Fprintf(&src, "// Code generated by %s; DO NOT EDIT.\n\n", generatedBy)
	fmt.Fprintf(&src, "package %s\n\n", pkg)
	src.WriteString(imports.render())
	src.Write(body.Bytes())

	formatted, err := format.Source(src.Bytes())
	if err != nil {
		// Emit unformatted source to aid debugging, then fail.
		_ = os.WriteFile(outPath, src.Bytes(), 0o644)
		return fmt.Errorf("formatting generated source: %w", err)
	}
	if dir := filepath.Dir(outPath); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(outPath, formatted, 0o644)
}

// emitSingle writes the var + builder for a one-model export.
func emitSingle(w *bytes.Buffer, imports *importSet, varName string, env *envelope) error {
	f, err := parseForest(env.Type, env.Model)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "// %s is the statically compiled %s\n", varName, env.Type)
	fmt.Fprintf(w, "// (%d trees, %d features, %d classes).\n", len(f.Trees), f.NFeatures, len(f.Classes))
	fmt.Fprintf(w, "var %s = build%s()\n\n", varName, varName)
	imports.ensemble, imports.tree = true, true
	imports.math = emitForest(w, "build"+varName, env.Type, f) || imports.math
	return nil
}

// emitBundle writes the var + builders for a multi-model bundle: one builder per
// model, the tuned metadata as raw JSON literals, and a goml.Bundle assembling
// them. Metadata stays raw so the numbers reach Go with the exact bits the
// export carried, and so a value go-ml has no typed accessor for still travels.
func emitBundle(w *bytes.Buffer, imports *importSet, varName string, env *envelope) error {
	if len(env.Models) == 0 {
		return fmt.Errorf("bundle has no models")
	}
	imports.goml, imports.ensemble, imports.tree = true, true, true

	names := make([]string, 0, len(env.Models))
	for name := range env.Models {
		names = append(names, name)
	}
	sort.Strings(names) // deterministic output: map order is not

	metaKeys := make([]string, 0, len(env.Metadata))
	for key := range env.Metadata {
		metaKeys = append(metaKeys, key)
	}
	sort.Strings(metaKeys)

	// Parse every model up front so the doc comment can describe them.
	forests := make(map[string]*forestJSON, len(names))
	for _, name := range names {
		f, err := parseForest(env.Models[name].Type, env.Models[name].Model)
		if err != nil {
			return fmt.Errorf("bundle model %q: %w", name, err)
		}
		forests[name] = f
	}

	fmt.Fprintf(w, "// %s is the statically compiled bundle.\n//\n// Models:\n", varName)
	for _, name := range names {
		f := forests[name]
		fmt.Fprintf(w, "//   - %s: %s, %d trees, %d features, %d classes\n",
			name, env.Models[name].Type, len(f.Trees), f.NFeatures, len(f.Classes))
	}
	if len(metaKeys) > 0 {
		fmt.Fprintf(w, "//\n// Metadata: %s\n", strings.Join(metaKeys, ", "))
	}
	fmt.Fprintf(w, "var %s = build%s()\n\n", varName, varName)

	fmt.Fprintf(w, "func build%s() *goml.Bundle {\n", varName)
	w.WriteString("\tmodels := map[string]goml.Model{\n")
	for _, name := range names {
		fmt.Fprintf(w, "\t\t%s: %s(),\n", strconv.Quote(name), builderName(varName, name))
	}
	w.WriteString("\t}\n")

	if len(metaKeys) > 0 {
		imports.json = true
		w.WriteString("\tmeta := map[string]json.RawMessage{\n")
		for _, key := range metaKeys {
			fmt.Fprintf(w, "\t\t%s: json.RawMessage(%s),\n",
				strconv.Quote(key), strconv.Quote(string(env.Metadata[key])))
		}
		w.WriteString("\t}\n")
	} else {
		w.WriteString("\tvar meta map[string]json.RawMessage\n")
		imports.json = true
	}

	fmt.Fprintf(w, "\tb, err := goml.NewBundle(models, meta)\n")
	w.WriteString("\tif err != nil {\n\t\tpanic(\"go-ml-gen: \" + err.Error())\n\t}\n")
	w.WriteString("\treturn b\n}\n\n")

	for _, name := range names {
		fmt.Fprintf(w, "// %s builds the %q model of %s.\n", builderName(varName, name), name, varName)
		imports.math = emitForest(w, builderName(varName, name), env.Models[name].Type, forests[name]) || imports.math
		w.WriteString("\n")
	}
	return nil
}

// builderName is the unexported builder for one bundle member, e.g. bundle
// "Cascade" model "screen" -> buildCascadeScreen. Anything that is not a letter
// or digit is dropped, so keys like "left-eye" or "stage 2" stay valid Go.
func builderName(varName, model string) string {
	var b strings.Builder
	b.WriteString("build")
	b.WriteString(varName)
	upper := true
	for _, r := range model {
		switch {
		case !unicode.IsLetter(r) && !unicode.IsDigit(r):
			upper = true
		case upper:
			b.WriteRune(unicode.ToUpper(r))
			upper = false
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func parseForest(typeName string, raw json.RawMessage) (*forestJSON, error) {
	switch typeName {
	case "RandomForestClassifier", "ExtraTreesClassifier":
		var f forestJSON
		if err := json.Unmarshal(raw, &f); err != nil {
			return nil, fmt.Errorf("parsing model: %w", err)
		}
		return &f, nil
	default:
		return nil, fmt.Errorf("unsupported model type %q", typeName)
	}
}

// emitForest writes the builder function for a tree ensemble of the given
// estimator type and reports whether it referenced the math package (for
// non-finite literals). The scikit-learn estimator name is also the name of the
// Go type and of its constructor's suffix, so it is used verbatim.
func emitForest(w *bytes.Buffer, funcName, typeName string, f *forestJSON) bool {
	var fw floatWriter
	fmt.Fprintf(w, "func %s() *ensemble.%s {\n", funcName, typeName)

	fmt.Fprintf(w, "\ttrees := []*tree.Tree{\n")
	for i := range f.Trees {
		t := &f.Trees[i]
		w.WriteString("\t\t{")
		writeInt32Slice(w, "Left", t.Left)
		writeInt32Slice(w, "Right", t.Right)
		writeInt32Slice(w, "Feature", t.Feature)
		w.WriteString("Threshold: []float64{")
		fw.writeFloats(w, t.Threshold)
		w.WriteString("}, ")
		writeBoolSlice(w, "MissingLeft", t.MissingLeft)
		w.WriteString("Value: []float64{")
		fw.writeFloats(w, t.Value)
		w.WriteString("}, ")
		fmt.Fprintf(w, "ValueWidth: %d},\n", t.ValueWidth)
	}
	w.WriteString("\t}\n")

	w.WriteString("\tclasses := []float64{")
	fw.writeFloats(w, f.Classes)
	w.WriteString("}\n")

	// Feature names travel with the model when the estimator was fitted on a
	// named frame. They are the caller's defence against feeding a retrained
	// model's columns in the old order, so they are compiled in too.
	opts := ""
	if len(f.FeatureNames) > 0 {
		w.WriteString("\tnames := []string{")
		for i, n := range f.FeatureNames {
			if i > 0 {
				w.WriteString(", ")
			}
			w.WriteString(strconv.Quote(n))
		}
		w.WriteString("}\n")
		opts = ", ensemble.WithFeatureNames(names)"
	}

	fmt.Fprintf(w, "\tm, err := ensemble.New%s(%d, classes, trees%s)\n", typeName, f.NFeatures, opts)
	w.WriteString("\tif err != nil {\n\t\tpanic(\"go-ml-gen: \" + err.Error())\n\t}\n")
	w.WriteString("\treturn m\n}\n")
	return fw.usedMath
}

func writeInt32Slice(w *bytes.Buffer, name string, xs []int32) {
	fmt.Fprintf(w, "%s: []int32{", name)
	for i, x := range xs {
		if i > 0 {
			w.WriteByte(',')
		}
		w.WriteString(strconv.FormatInt(int64(x), 10))
	}
	w.WriteString("}, ")
}

func writeBoolSlice(w *bytes.Buffer, name string, xs []bool) {
	fmt.Fprintf(w, "%s: []bool{", name)
	for i, x := range xs {
		if i > 0 {
			w.WriteByte(',')
		}
		if x {
			w.WriteString("true")
		} else {
			w.WriteString("false")
		}
	}
	w.WriteString("}, ")
}

// floatWriter formats float64 literals that round-trip exactly, emitting
// math.Inf/math.NaN for non-finite values and recording that math was used.
type floatWriter struct{ usedMath bool }

func (fw *floatWriter) writeFloats(w *bytes.Buffer, xs []float64) {
	for i, x := range xs {
		if i > 0 {
			w.WriteByte(',')
		}
		w.WriteString(fw.format(x))
	}
}

func (fw *floatWriter) format(x float64) string {
	switch {
	case math.IsInf(x, 1):
		fw.usedMath = true
		return "math.Inf(1)"
	case math.IsInf(x, -1):
		fw.usedMath = true
		return "math.Inf(-1)"
	case math.IsNaN(x):
		fw.usedMath = true
		return "math.NaN()"
	default:
		// Shortest representation that parses back to the identical float64.
		return strconv.FormatFloat(x, 'g', -1, 64)
	}
}

// exportVarName turns a file base name into an exported Go identifier, e.g.
// "forest_bench.json" -> "ForestBench".
func exportVarName(path string) string {
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	var b strings.Builder
	upper := true
	for _, r := range base {
		switch {
		case r == '_' || r == '-' || r == '.' || r == ' ':
			upper = true
		case upper:
			b.WriteRune(unicode.ToUpper(r))
			upper = false
		default:
			b.WriteRune(r)
		}
	}
	name := b.String()
	if name == "" || !unicode.IsLetter(rune(name[0])) {
		name = "Model" + name
	}
	return name
}
