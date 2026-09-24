# Example: U=10 (Ng=2)

`R.in` and `L.in` are the converged (R, lambda) of the independent reference
implementation at U/t=10, one shared row/block for both fragments (the
converged solution is symmetric, R0 = R1). They are used as the initial guess.

Run from this directory:

```
python3 ../../gga_dimer.py --U 10.0 --Ng 2
```

Expected result (matches the reference to 6 significant figures; the seed is
already the converged solution, so it converges in 1 iteration):

```
Ng=2  U/t=10.0000  Z0=0.991894  Z1=0.991894  E_var/t=-10.101314  docc=0.005204  iters=1  converged=True
```

At this solution `Delta_qp` and `Delta_new` (printed at the end of every run)
agree (`max|Delta_qp - Delta_new| ~ 9e-6`), both with eigenvalues
`{0.002687, 0.5, 0.997313}`: unlike the U=2 "forward" example, this is a
ghost-assisted solution with active ghost orbitals.

## Direct minimization (no impurity problem)

The same seed can be used with `not_imp_gga_dimer.py`
(see `doc/not_impurity.tex`):

```
PYTHONPATH=../.. python3 ../../not_imp_gga_dimer.py --U 10 --Ng 2
```

Expected: `E_var/t = -10.10131406`, constraints satisfied to ~5e-15, KKT
residuals ~1e-7 or smaller, quasiparticle energies
`eqp = [-5.286, -4.294, ~0, ~0, 4.294, 5.286]` (the middle pair is degenerate at
the Fermi level; its exact value, ~1e-11 here, depends on the minimum-norm choice for lambda). Note that from random
starts (no `R.in`/`L.in`) that code only reaches the higher inert-ghost
solution `E_var/t = -10.0`; the seed is what selects the lower solution.

## Spectral function (optional)

Both codes accept `--omega`, `--eta`, `--domega` (all three, or none) and then write
`A_omega.txt` (columns `omega`, `A(omega)`):

```
python3 ../../gga_dimer.py --U 10.0 --Ng 2 --omega 15 --eta 0.05 --domega 0.05
PYTHONPATH=../.. python3 ../../not_imp_gga_dimer.py --U 10 --Ng 2 --omega 15 --eta 0.05 --domega 0.05
```

For this seed both give `integral A d(omega) = 3.96` (sum rule 4 = 2 sites x 2 spins) and the
two `A(omega)` agree to ~6e-5.
