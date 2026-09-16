# Example: Ng=2, U=10

`R.in` and `L.in` are the converged (R, lambda) from an independent reference
implementation, used here as an initial guess to validate `gga_dimer.py`.

Run from this directory:

```
python3 ../../gga_dimer.py --U 10.0 --Ng 2
```

Expected result (matches the reference to 6 significant figures):

```
Ng=2  U/t=10.0000  Z=R^2=0.991895  E_var/t=-10.101316  docc=0.005204
```

## Spectral function (optional)

Add `--omega`, `--eta`, `--domega` (all three, or none) to also compute the
quasiparticle spectral function `A(omega)` from the converged `H_qp` and
write it to `A_omega.txt` (see `doc/code.tex`, Section "Post-processing:
quasiparticle Green's function and spectral function"):

```
python3 ../../gga_dimer.py --U 10.0 --Ng 2 --omega 15 --eta 0.05 --domega 0.05
```

`A_omega.txt` has two columns, `omega` and `A(omega)`, on a uniform grid from
`-15` to `+15` in steps of `0.05`. As a normalization check,
`\int A(omega) d(omega) ~= 3.96`, close to the expected sum-rule value of 4
(2 sites x 2 spins); the small deficit is from the finite frequency cutoff
and the Lorentzian tails extending past `+-omega`.
