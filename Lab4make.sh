#file : Thierstein_Emily_HW_3.py

#python Thierstein_Emily_HW_3.py

Lab4out=
TEX=pdflatex
BIBTEX=bibtex
BUILDTEX=$(TEX) $(Lab4out).tex

all:
    $(BUILDTEX)
    $(BIBTEX) $(Lab4out)
    $(BUILDTEX)
    $(BUILDTEX)
clean-all:
    rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~

clean:
    rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~