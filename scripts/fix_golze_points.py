"""Recompute the poor points of the V = 1/sqrt2, sqrt2 energy data (B_c=3 constraint < 1e-9, KKT < 1e-5, E-E_ED < 1.2e-7;
B_c=1 loop converged) from random starts / a different mixing; patch cederbaum_energies_V<V>.npz. One log line per attempt."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe

for V in pe.VS:
    d = dict(np.load(pe.fileV(V)))
    for tc in pe.TCS:
        key = f"V{V}_tc{tc}"; r = d[key]
        E_ed = r[0]
        bad3 = not (r[7] < 1e-9 and r[8] < 1e-5 and r[4] - E_ed < 1.2e-7)
        bad1 = r[6] < 0.5
        if bad3:
            m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m); rng = np.random.default_rng(int(1000 * V + 7000 * tc) + 11)
            best = None
            for k in range(40):
                res = N.minimize_once(P, N.random_start(P, rng), maxiter=400)
                cons = np.abs(P.constraints(res.x)).max()
                o = N.recover(P, res.x) if cons < 1e-9 else None
                print(f"  {key} B3 try {k}: dE {res.fun-E_ed:+.2e} cons {cons:.0e} kkt {o['kkt'] if o else float('nan'):.0e}", flush=True)
                if o is not None and o["kkt"] < 1e-5 and (best is None or res.fun < best[0].fun):
                    best = (res, o)
                if best is not None and best[0].fun - E_ed < 1.2e-7:
                    break
            if best is not None:
                res, o = best
                d[key] = np.array([r[0], r[1], r[2], r[3], res.fun, o["nu"], r[6], np.abs(P.constraints(res.x)).max(), o["kkt"]])
                print(f"{key}: B3 fixed dE {res.fun-E_ed:+.2e} kkt {o['kkt']:.0e}", flush=True)
            else:
                print(f"{key}: B3 NOT fixed", flush=True)
        if bad1:
            for mix, it in ((0.3, 6000), (0.2, 10000), (0.1, 20000)):
                s1 = g.run(1.0, 10.0, V, tc, 1, max_iter=it, mix=mix, tol=1e-11)
                print(f"  {key} B1 mix {mix}: conv {s1['converged']} dE {s1['E_var']-E_ed:+.2e}", flush=True)
                if s1["converged"]:
                    d[key] = np.concatenate([d[key][:2], [s1["E_var"], s1["nu"]], d[key][4:6], [1.0], d[key][7:]]); break
    np.savez(pe.fileV(V), **d)
print("done")
