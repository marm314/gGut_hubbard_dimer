#!/bin/sh
# Resume/reproduce U/t=1.3000, Ng=2, deltaV/t=2 from this directory's
# R.in/L.in. Run from anywhere -- both the solver script and the guess files are
# referenced by absolute/directory-relative path below.
cd "$(dirname "$0")"
/usr/bin/python3 /home/marm314/Installers/gGut_mau/gga_dimer.py --U 1.3 --Ng 2 --deltaV 2 --R-file R.in --lam-file L.in
