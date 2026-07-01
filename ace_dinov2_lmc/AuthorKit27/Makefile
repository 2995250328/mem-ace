TEX := ace_lmc_submission
TINY_BIN := /data/xwh/.TinyTeX/bin/x86_64-linux
LATEXMK := $(if $(wildcard $(TINY_BIN)/latexmk),$(TINY_BIN)/latexmk,latexmk)
TEXPATH := $(if $(wildcard $(TINY_BIN)/pdflatex),$(TINY_BIN):$(PATH),$(PATH))

.PHONY: all clean

all:
	PATH="$(TEXPATH)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX).tex

clean:
	PATH="$(TEXPATH)" $(LATEXMK) -C $(TEX).tex
