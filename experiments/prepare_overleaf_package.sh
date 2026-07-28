#!/bin/bash
# Prepare Overleaf package for ProCyon v2
set -e

PROJECT_DIR="/hd/liujx/microbiome_llm_project"
PKG_DIR="$PROJECT_DIR/ProCyon_v2/overleaf_package"
ANALYSIS="$PROJECT_DIR/ProCyon_v2/analysis"

echo "=== ProCyon v2 Overleaf Package Builder ==="

# 1. Create package directory
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/figures" "$PKG_DIR/tables"

# 2. Copy main files
cp "$ANALYSIS/main.tex" "$PKG_DIR/"
cp "$ANALYSIS/references.bib" "$PKG_DIR/"

# 3. Copy all figures
echo "Copying figures..."
for fig in dataset_architecture_figure week1_figure \
           shap_reliability calibration_error_cases \
           inductive_bias_figure paper_figure_shap_analysis \
           phase1_representation_analysis phase35_heterogeneity \
           phase36_robustness phase3_biological_validation \
           umap_visualization; do
    src="$ANALYSIS/${fig}.png"
    if [ -f "$src" ]; then
        cp "$src" "$PKG_DIR/figures/"
        echo "  + figures/${fig}.png"
    fi
done

# 4. Copy LaTeX table fragments
echo "Copying tables..."
for tex in overleaf_tables overleaf_tables_standalone week1_tables \
           calibration_tables structural_baselines_table; do
    src="$ANALYSIS/${tex}.tex"
    if [ -f "$src" ]; then
        cp "$src" "$PKG_DIR/tables/"
        echo "  + tables/${tex}.tex"
    fi
done

# 5. SHAP residue check
echo ""
echo "=== SHAP Residue Check ==="
if grep -qi "shap" "$PKG_DIR/main.tex"; then
    echo "WARNING: SHAP references still found in main.tex:"
    grep -in "shap" "$PKG_DIR/main.tex"
else
    echo "PASS: No SHAP references in main.tex"
fi

# 6. Check \includegraphics references
echo ""
echo "=== Figure Reference Check ==="
missing=0
for fig in $(grep -oP 'includegraphics\{[^}]*\}' "$PKG_DIR/main.tex" | sed 's/includegraphics{//;s/}//'); do
    if [ ! -f "$PKG_DIR/$fig" ]; then
        echo "MISSING: $fig"
        missing=1
    fi
done
[ $missing -eq 0 ] && echo "PASS: All figure references resolved"

# 7. Check \cite references
echo ""
echo "=== Citation Check ==="
if [ -f "$PKG_DIR/references.bib" ]; then
    missing_cite=0
    for cite in $(grep -oP '\\cite\{[^}]*\}' "$PKG_DIR/main.tex" | sed 's/\\cite{//;s/}//' | tr ',' '\n'); do
        if ! grep -q "@.*{$cite," "$PKG_DIR/references.bib"; then
            echo "MISSING citation: $cite"
            missing_cite=1
        fi
    done
    [ $missing_cite -eq 0 ] && echo "PASS: All citations in references.bib"
else
    echo "WARNING: references.bib not found"
fi

# 8. Try LaTeX compilation
echo ""
echo "=== LaTeX Compilation Check ==="
cd "$PKG_DIR"
if command -v pdflatex &>/dev/null; then
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    if [ -f main.pdf ]; then
        echo "PASS: main.tex compiles successfully ($(wc -c < main.pdf) bytes)"
        rm -f main.aux main.log main.out main.pdf
    else
        echo "WARNING: Compilation produced no PDF (check main.log)"
    fi
else
    echo "SKIP: pdflatex not available"
fi

# 9. Generate zip
ZIP_NAME="ProCyon_v2_overleaf_package.zip"
cd "$PKG_DIR/.."
rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" overleaf_package/ > /dev/null
echo ""
echo "=== Package Ready ==="
echo "  $(du -h $ZIP_NAME | cut -f1)  $PROJECT_DIR/ProCyon_v2/$ZIP_NAME"

# 10. File manifest
echo ""
echo "=== File Manifest ==="
find "$PKG_DIR" -type f | sort | while read f; do
    echo "  $(du -h "$f" | cut -f1)  ${f#$PROJECT_DIR/ProCyon_v2/}"
done

# 11. Git push if requested
if [ "$1" = "--push" ]; then
    echo ""
    echo "=== Pushing to GitHub ==="
    cd "$PROJECT_DIR"
    git add ProCyon_v2/overleaf_package/ ProCyon_v2/ProCyon_v2_overleaf_package.zip experiments/prepare_overleaf_package.sh
    git commit -m "Overleaf package: compilable main.tex + all figures + references.bib + zip" || echo "(nothing to commit)"
    git push origin main
fi

echo ""
echo "DONE. Upload $PROJECT_DIR/ProCyon_v2/ProCyon_v2_overleaf_package.zip to Overleaf."
