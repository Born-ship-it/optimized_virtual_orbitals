# .latexmkrc – latexmk configuration for the thesis
# Run with:  latexmk -pdf main.tex

$pdf_mode        = 1;           # Use pdflatex
$pdflatex        = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use      = 2;           # Run bibtex/biber automatically
$clean_ext       = 'synctex.gz synctex(busy) run.xml tex.bak bbl bcf fdb_latexmk aux toc lof lot out';
$out_dir         = 'build';     # Place compiled output in build/
