"""Do ghosts on o and v change the picture? Direct minimization (not_imp_gga_cederbaum) with (B_c, B_o, B_v) generalized.
Per start: log line; the best (lowest energy with constraint residual < 1e-9) solution is saved to cederbaum_data/bobv_<tag>.npz.
python3 test_bo_bv.py Bc Bo Bv V tc nstarts seed"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import exact_cederbaum as ed

Bc, Bo, Bv = map(int, sys.argv[1:4]); V, tc = float(sys.argv[4]), float(sys.argv[5]); ns, seed = int(sys.argv[6]), int(sys.argv[7])
m = g.Model(1.0, 10.0, V, tc, Bc, 1e-9, "cv", Bo=Bo, Bv=Bv); P = N.Problem(m)
rng = np.random.default_rng(seed)
r = ed.solve(1.0, 10.0, V, tc, ht="cv")
tag = f"B{Bc}{Bo}{Bv}_V{V:.4f}_tc{tc}"
print(f"{tag}: ED E0={r['E0']:.9f}, nz={P.nx + P.n * P.N}, ncons={P.ncons}", flush=True)
best = None
for k in range(ns):
    res = N.minimize_once(P, N.random_start(P, rng))
    c = np.abs(P.constraints(res.x)).max()
    out = N.recover(P, res.x)
    print(f"start {k}: E-E_ED={res.fun - r['E0']:+.3e} cons={c:.1e} kkt={out['kkt']:.1e} comm={out['comm']:.1e} nu_c={out['nu']:.6f} it={res.nit} "
          f"poles {np.round(out['poles'], 3)} w {np.array2string(out['weights'], precision=1)}", flush=True)
    if c < 1e-9 and (best is None or res.fun < best[0]):
        best = (res.fun, res.x, out)
        np.savez(f"cederbaum_data/bobv_{tag}.npz", z=res.x, E=res.fun, EED=r["E0"], poles=out["poles"], w=out["weights"], eps=out["eps"], Hqp=out["Hqp"],
                 R0=out["R"][0], Ro=out["R"][1], Rv=out["R"][2])
print("best", None if best is None else f"E-E_ED={best[0]-r['E0']:+.3e}")
