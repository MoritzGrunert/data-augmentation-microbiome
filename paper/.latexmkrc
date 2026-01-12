# ----------------------------
# Main document
# ----------------------------
@default_files = ('main.tex');

# ----------------------------
# Output directory
# ----------------------------
$out_dir = 'build';
$aux_dir = 'build';

# Ensure build/ exists (latexmk will call this before running LaTeX)
$ensure_path = 1;

# ----------------------------
# Build mode
# ----------------------------
$pdf_mode = 1;  # tex -> pdf

# ----------------------------
# LaTeX engine options
# ----------------------------
$pdflatex = 'pdflatex -8bit -file-line-error -halt-on-error --shell-escape -synctex=1 %O %S';

# ----------------------------
# Cleanup configuration
# ----------------------------
# Extra extensions cleaned by "latexmk -c" / "latexmk -C"
$clean_ext .= " synctex.gz run.xml %R-blx.bib lol bbl";
$cleanup_includes_cusdep_generated = 1;

# Uncomment to list dependents after compilation
# $dependents_list = 1;

# ----------------------------
# Custom dependency: SVG -> PDF (via Inkscape)
# ----------------------------
add_cus_dep('svg', 'pdf', 0, 'svg2pdf');
sub svg2pdf {
    my ($base) = @_;
    # $base is the file path without extension
    my $src_svg = "$base.svg";
    my $out_pdf = "$out_dir/$base.pdf";

    # Create subdirectories in build/ if needed
    my $out_dirname = $out_pdf;
    $out_dirname =~ s{/[^/]+$}{};
    system("mkdir", "-p", $out_dirname);

    return system("inkscape",
        "--export-text-to-path",
        "--export-area-drawing",
        "--export-type=pdf",
        "--export-filename=$out_pdf",
        $src_svg
    );
}

# ----------------------------
# Custom dependency: makeindex
# ----------------------------
add_cus_dep('idx', 'ind', 0, 'makeindex_run');
sub makeindex_run {
    my ($base) = @_;
    # idx will be in build/ because aux_dir=build
    return system("makeindex \"$aux_dir/$base.idx\"");
}
push @generated_exts, 'ind';