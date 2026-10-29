"""B_c=5 data on the grids of the energy and spectra figures (Delta=1, E_c=10, c-v hopping).
For each (V, t_c): B_c=3 solution (random starts until E-E_ED < 1.2e-7 and KKT < 1e-5), embedded into B_c=5 with the extra
core ghosts asleep, NSEED perturbed runs (noises 0.03, 0.1) -> keep the lowest accepted one (constraints < 1e-9, KKT < 1e-5);
also NRAND random B_c=5 starts to look for anything lower. One process per V; one log line per point.
Usage: python3 scripts/compute_bc5_data.py V   -> cederbaum_bc5_V<V>.npz"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
from plot_cederbaum_energies import TCS

NOISES = [0.03, 0.1]
NSEED = 4
NRAND = 6


def accepted(P, res):
    cons = np.abs(P.constraints(res.x)).max()
    if cons > 1e-9:
        return None
    out = N.recover(P, res.x)
    return out if out["kkt"] < 1e-5 else None


if __name__ == "__main__":
    V = float(sys.argv[1]); data = {}; t_start = time.time()
    zprev = None
    for tc in TCS:
        r = ed.solve(1.0, 10.0, V, tc, ht="cv")
        m3 = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P3 = N.Problem(m3); rng = np.random.default_rng(int(1000 * V + 7000 * tc) + 1)
        best3 = None
        for k in range(30):
            res = N.minimize_once(P3, N.random_start(P3, rng), maxiter=400)
            if accepted(P3, res) is not None and (best3 is None or res.fun < best3.fun):
                best3 = res
            if best3 is not None and best3.fun - r["E0"] < 1.2e-7:
                break
        if best3 is None:
            print(f"V={V} t_c={tc}: no B_c=3 solution", flush=True); continue
        P5, z0 = embed(best3.x, V, tc)
        runs = []
        for nz in NOISES:
            for k in range(NSEED):
                z = z0 + nz * np.random.default_rng(31 * k + int(1000 * nz) + int(100 * tc)).standard_normal(z0.size)
                res = N.minimize_once(P5, z, maxiter=400); out = accepted(P5, res)
                if out is not None: runs.append((res.fun, out, res))
        rr = np.random.default_rng(int(1000 * V + 7000 * tc) + 5); nrand_ok = 0; rand_best = None
        for k in range(NRAND):
            res = N.minimize_once(P5, N.random_start(P5, rr), maxiter=400); out = accepted(P5, res)
            if out is not None:
                nrand_ok += 1; runs.append((res.fun, out, res)); rand_best = res.fun if rand_best is None else min(rand_best, res.fun)
        if not runs:
            print(f"V={V} t_c={tc}: no converged B_c=5 run", flush=True); continue
        runs.sort(key=lambda x: x[0]); E5, o5, res5 = runs[0]
        key = f"V{V}_tc{tc}"
        data[key + "_E"] = np.array([r["E0"], r["nu_c"], best3.fun, N.recover(P3, best3.x)["nu"], E5, o5["nu"], o5["kkt"], len(runs),
                                     runs[-1][0] - runs[0][0], nrand_ok])
        data[key + "_poles"] = o5["poles"]; data[key + "_w"] = o5["weights"]
        print(f"[{time.time()-t_start:5.0f}s] V={V} t_c={tc}: B3 dE={best3.fun-r['E0']:+.2e} | B5 dE={E5-r['E0']:+.2e} (B5-B3={E5-best3.fun:+.1e}) "
              f"nu-nu_ED={o5['nu']-r['nu_c']:+.1e} KKT={o5['kkt']:.0e} converged {len(runs)}/{2*NSEED+NRAND} (random {nrand_ok}/{NRAND}) "
              f"spread {runs[-1][0]-runs[0][0]:.1e} main pole {o5['poles'][np.argmax(o5['weights'])]:.3f} w={o5['weights'].max():.4f}", flush=True)
    np.savez(f"cederbaum_data/cederbaum_bc5_V{V}.npz", **data)
    print("saved", flush=True)
