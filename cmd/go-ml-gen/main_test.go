package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modelsDir is the repository's export corpus, three directories up.
const modelsDir = "../../testdata/models"

// declsOf parses generated source and returns the top-level names it declares:
// "var X", "func f". Anything the generator emits has to parse, and these tests
// assert on the shape rather than on the exact bytes.
func declsOf(t *testing.T, path string) map[string]bool {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := parser.ParseFile(token.NewFileSet(), path, src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("generated source does not parse: %v", err)
	}

	decls := map[string]bool{}
	for _, d := range file.Decls {
		switch d := d.(type) {
		case *ast.FuncDecl:
			decls["func "+d.Name.Name] = true
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				if v, ok := spec.(*ast.ValueSpec); ok {
					for _, name := range v.Names {
						decls["var "+name.Name] = true
					}
				}
			}
		}
	}
	return decls
}

func TestGenerateSingleModel(t *testing.T) {
	out := filepath.Join(t.TempDir(), "iris_gen.go")
	if err := generate(filepath.Join(modelsDir, "iris.json"), out, "models", "Iris"); err != nil {
		t.Fatal(err)
	}

	decls := declsOf(t, out)
	for _, want := range []string{"var Iris", "func buildIris", "func IrisAvailable"} {
		if !decls[want] {
			t.Errorf("generated file is missing %s (has %v)", want, decls)
		}
	}

	src, _ := os.ReadFile(out)
	// Feature names are part of the model's contract, so they must survive
	// compilation rather than being dropped on the way in.
	if !strings.Contains(string(src), "WithFeatureNames") {
		t.Error("an export with feature names should compile them in")
	}
	if !strings.Contains(string(src), "func IrisAvailable() bool { return true }") {
		t.Error("a real model's Available must report true")
	}
}

func TestGenerateBundle(t *testing.T) {
	out := filepath.Join(t.TempDir(), "bundle_gen.go")
	if err := generate(filepath.Join(modelsDir, "iris_bundle.json"), out, "models", "Cascade"); err != nil {
		t.Fatal(err)
	}

	decls := declsOf(t, out)
	for _, want := range []string{
		"var Cascade", "func buildCascade", "func CascadeAvailable",
		"func buildCascadeScreen", "func buildCascadeConfirm",
	} {
		if !decls[want] {
			t.Errorf("generated bundle is missing %s (has %v)", want, decls)
		}
	}
	if src, _ := os.ReadFile(out); !strings.Contains(string(src), "goml.NewBundle") {
		t.Error("a bundle should be assembled with goml.NewBundle")
	}
}

// TestStubDeclaresTheSameNames is the contract that makes the placeholder
// useful: a call site written against the stub keeps compiling, unchanged, once
// a real export replaces it.
func TestStubDeclaresTheSameNames(t *testing.T) {
	dir := t.TempDir()
	stub := filepath.Join(dir, "stub_gen.go")
	real := filepath.Join(dir, "real_gen.go")

	if err := generateStub(stub, "models", "Iris", typeRandomForest, nil); err != nil {
		t.Fatal(err)
	}
	if err := generate(filepath.Join(modelsDir, "iris.json"), real, "models", "Iris"); err != nil {
		t.Fatal(err)
	}

	stubDecls, realDecls := declsOf(t, stub), declsOf(t, real)
	for _, name := range []string{"var Iris", "func IrisAvailable"} {
		if !stubDecls[name] || !realDecls[name] {
			t.Errorf("%s: stub=%v real=%v", name, stubDecls[name], realDecls[name])
		}
	}
	if src, _ := os.ReadFile(stub); !strings.Contains(string(src), "func IrisAvailable() bool { return false }") {
		t.Error("a placeholder's Available must report false")
	}
}

func TestStubTypes(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct{ kind, wantType string }{
		{typeRandomForest, "*ensemble.RandomForestClassifier"},
		{typeExtraTrees, "*ensemble.ExtraTreesClassifier"},
		{typeBundle, "*goml.Bundle"},
	} {
		out := filepath.Join(dir, tc.kind+"_gen.go")
		if err := generateStub(out, "models", "M", tc.kind, nil); err != nil {
			t.Fatalf("%s: %v", tc.kind, err)
		}
		src, _ := os.ReadFile(out)
		if !strings.Contains(string(src), "var M "+tc.wantType) {
			t.Errorf("%s: want a %s var, got:\n%s", tc.kind, tc.wantType, src)
		}
	}
}

func TestStubRejectsBadInvocations(t *testing.T) {
	out := filepath.Join(t.TempDir(), "x.go")
	cases := map[string]error{
		"no var name":     generateStub(out, "models", "", typeRandomForest, nil),
		"no output":       generateStub("", "models", "M", typeRandomForest, nil),
		"unknown type":    generateStub(out, "models", "M", "GradientBoosting", nil),
		"input passed in": generateStub(out, "models", "M", typeRandomForest, []string{"model.json"}),
	}
	for name, err := range cases {
		if err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestGenerateRejectsUnsupportedDocuments(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"unknown format": `{"format":"go-ml/v999","type":"RandomForestClassifier","model":{}}`,
		"unknown type":   `{"format":"go-ml/v1","type":"HistGradientBoosting","model":{}}`,
		"empty bundle":   `{"format":"go-ml/bundle-v1","models":{}}`,
		"not json":       `{`,
	}
	for name, doc := range cases {
		in := filepath.Join(dir, "in.json")
		if err := os.WriteFile(in, []byte(doc), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := generate(in, filepath.Join(dir, "out.go"), "models", "M"); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestExportVarName(t *testing.T) {
	for in, want := range map[string]string{
		"iris.json":                "Iris",
		"forest_bench.json":        "ForestBench",
		"extratrees-balanced.json": "ExtratreesBalanced",
		// A name that cannot start an identifier gets the "Model" prefix.
		"/tmp/2024_model.json": "Model2024Model",
	} {
		if got := exportVarName(in); got != want {
			t.Errorf("exportVarName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBuilderName(t *testing.T) {
	for _, tc := range []struct{ bundle, model, want string }{
		{"Cascade", "screen", "buildCascadeScreen"},
		{"Cascade", "left-eye", "buildCascadeLeftEye"},
		{"Cascade", "stage 2", "buildCascadeStage2"},
	} {
		if got := builderName(tc.bundle, tc.model); got != tc.want {
			t.Errorf("builderName(%q, %q) = %q, want %q", tc.bundle, tc.model, got, tc.want)
		}
	}
}
