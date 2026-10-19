#!/bin/sh
# Resume/reproduce the CONFIRMED WINNING solution at U/t=2.7000, Ng=4,
# deltaV/t=2 from this directory's R.in/L.in, with either solver.
# Run from anywhere -- both solver scripts and the guess files are referenced
# by absolute/directory-relative path below.
cd "$(dirname "$0")"
/usr/bin/python3 /home/marm314/Installers/gGut_mau/gga_dimer.py --U 2.7 --Ng 4 --deltaV 2 --R-file R.in --lam-file L.in
/usr/bin/python3 /home/marm314/Installers/gGut_mau/not_imp_gga_dimer.py --U 2.7 --Ng 4 --deltaV 2 --R-file R.in --lam-file L.in
