# Example: Ng=2, U=2

`R.in` and `L.in` are the iteration-0 guess from the reference implementation's
log ($HOME/2.0.out), one row per atom (the two fragments are NOT symmetric in
this case -- atom 0 and atom 1 have genuinely different R and lambda).

Run from this directory:

```
python3 ../../gga_dimer.py --U 2.0 --Ng 2
```

Reference converged value (from 2.0.out, iteration 4): `Evar = -3.12566`.

Status: converges to `E_var/t = -3.125656`, matching the reference to
5-6 significant figures, with the genuinely asymmetric fixed point
(atom 0 and atom 1 keep distinct R/lambda; Z0=0.999912, Z1=0.999913).

## Spectral function (optional)

Add `--omega`, `--eta`, `--domega` (all three, or none) to also compute the
quasiparticle spectral function `A(omega)` from the converged `H_qp` and
write it to `A_omega.txt` (see `doc/code.tex`, Section "Post-processing:
quasiparticle Green's function and spectral function"):

```
python3 ../../gga_dimer.py --U 2.0 --Ng 2 --omega 10 --eta 0.05 --domega 0.05
```

`A_omega.txt` has two columns, `omega` and `A(omega)`, on a uniform grid from
`-10` to `+10` in steps of `0.05`. As a normalization check,
`\int A(omega) d(omega) ~= 3.98`, close to the expected sum-rule value of 4
(2 sites x 2 spins).
