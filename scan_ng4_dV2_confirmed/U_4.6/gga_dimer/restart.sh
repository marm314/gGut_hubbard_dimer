#!/bin/sh
# Resume/reproduce U/t=4.6000, Ng=4, deltaV/t=2 from this directory's
# R.in/L.in. Run from anywhere -- both the solver script and the guess files are
# referenced by absolute/directory-relative path below.
cd "$(dirname "$0")"
/usr/bin/python3 /home/marm314/Installers/gGut_mau/gga_dimer.py --U 4.6 --Ng 4 --deltaV 2 --R-file R.in --lam-file L.in
