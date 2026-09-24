# Example: "forward" (Ng=2, U=2)

`R.in`/`L.in` are the seed used for one of the two independent-code
validation runs of `gga_dimer.py` against the reference C++ implementation
(see `doc/code.tex`, Sections "Numerical solution of the self-consistency
loop" and "Are the converged points solutions?").

Run from this directory:

```
python3 ../../gga_dimer.py --U 2.0 --Ng 2
```

Expected result: converges in 5 iterations to `E_var/t = -3.125000`
(Z0 = Z1 = 0.9375) -- the plain-Gutzwiller solution, with both ghost
orbitals decoupled (inert). This matches the reference C++ run on the same
seed to the digits printed in its log.

Note: at this solution, `Delta_qp` and `Delta_new` (printed automatically at
the end of every run) share the same eigenvalue spectrum per fragment
(`{0, 0.5, 1}`) but are *not* the same matrix (`max|Delta_qp - Delta_new| ~
0.97`): the two inert ghost orbitals have their occupations swapped between
the quasiparticle side and the impurity side. See `doc/code.tex` for what
this means.
