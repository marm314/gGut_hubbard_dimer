# gGut_hubbard_dimer

Ghost Gutzwiller Approximation (gGA) for the half-filled two-site Hubbard
dimer, with a tunable number of ghost bath orbitals `Ng`, implemented in
Python and benchmarked against exact diagonalization and an independent
reference (C++) implementation.

## Contents

- `gga_dimer.py` -- the gGA self-consistency solver (two independent
  fragments, analytic-Jacobian Newton solve for lambda, quasiparticle
  Green's function / spectral function post-processing). Run
  `python3 gga_dimer.py --help` for CLI options.
- `exact_dimer.py` -- exact diagonalization of the 2-site Hubbard dimer,
  used as ground truth.
- `examples/Ng2_U2/`, `examples/Ng2_U10/` -- worked examples with
  reference-seeded initial guesses (`R.in`, `L.in`) and expected results
  (see each directory's `README.md`).
- `doc/code.tex` (compiled: `doc/code.pdf`) -- a pedantic write-up of the
  theory (the constrained variational Lagrangian), the equations actually
  implemented, the numerical algorithm, and the CLI/file conventions.

## Quick start

```
python3 gga_dimer.py --U 10.0 --Ng 2
```

or, from one of the example directories, seeded with a known-good guess:

```
cd examples/Ng2_U10
python3 ../../gga_dimer.py --U 10.0 --Ng 2
```
