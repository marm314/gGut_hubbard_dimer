# Example: "backward" (Ng=2, U=2)

`R.in`/`L.in` are the seed used for the other of the two independent-code
validation runs of `gga_dimer.py` against the reference C++ implementation
(see `doc/code.tex`, Sections "Numerical solution of the self-consistency
loop" and "Are the converged points solutions?").

Run from this directory:

```
python3 ../../gga_dimer.py --U 2.0 --Ng 2
```

Expected result: converges in 5 iterations to `E_var/t = -3.125661`
(Z0 = Z1 = 0.999911) -- a genuinely ghost-assisted, asymmetric solution
(atom 0 and atom 1 keep distinct R/lambda). This matches the reference C++
run on the same seed to the digits printed in its log, and the final R and
lambda agree with the reference to ~1e-9.

Note: at this solution, `Delta_qp` and `Delta_new` (printed automatically at
the end of every run) agree, both in eigenvalue spectrum and as full
matrices (`max|Delta_qp - Delta_new| ~ 4e-9`): unlike the "forward" example,
this is a genuine stationary point of the underlying Lagrangian (see
`check_stationarity.py`, `check_hessian.py` and `doc/code.tex`).
