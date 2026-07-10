# Mem-ACE AAAI Draft

This directory contains an initial AAAI-style LaTeX source draft for Mem-ACE.

## Files

- `main.tex`: manuscript entry point.
- `sections/`: abstract, introduction, related work, method, experiments, discussion, conclusion.
- `tables/`: current paper-safe and placeholder result tables.
- `references.bib`: citation placeholders that must be verified against official BibTeX/local PDFs.
- `figures/`: reserved for method and result figures.

## Compile

Place the official AAAI style file (`aaai26.sty`, `aaai25.sty`, or `aaai.sty`) in this directory, then run:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Evidence Rules

Do not submit the current draft as final. It intentionally contains TODO placeholders for:

- clean all-scene Indoor6 ACE-FCN+GLACE aggregation;
- full Wayspots scene aggregation;
- fixed-global protocol consolidation across all paper-facing rows;
- zero/random/shuffled memory controls;
- latent token capacity sweep;
- efficiency table;
- Cambridge, APR+LMC, and ACE-G+LMC extensions;
- official BibTeX verification.

The current prose frames Mem-ACE as fixed-global compact scene-memory conditioning and avoids claiming universal improvement.
