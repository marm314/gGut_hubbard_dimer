"""Fill (V, t_c) points missing from cederbaum_bc5_V<V>.npz (no B_c=3 seed from random starts) by continuation:
B_c=3 solution at the next larger t_c of the grid (random starts there), then B_c=3 at the missing t_c seeded from it with
small noise, then the B_c=5 step exactly as in compute_bc5_data.py."""
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
from plot_cederbaum_energies import TCS, VS
from compute_bc5_data import accepted, NOISES, NSEED, NRAND


def bc3_random(V, tc, ntry=40, seed=0):
    m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m); rng = np.random.default_rng(seed + int(1000 * V + 7000 * tc))
    r = ed.solve(1.0, 10.0, V, tc, ht="cv"); best = None
    for k in range(ntry):
        res = N.minimize_once(P, N.random_start(P, rng), maxiter=400)
        if accepted(P, res) is not None and (best is None or res.fun < best.fun):
            best = res
        if best is not None and best.fun - r["E0"] < 1.2e-7:
            break
    return best


def bc3_chain(V, tc):
    """B_c=3 solution at tc by continuation from larger t_c values (random starts at the first one that works)."""
    larger = [t for t in TCS if t > tc]
    m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m); r = ed.solve(1.0, 10.0, V, tc, ht="cv")
    for t0 in sorted(larger):                     # nearest larger first
        z = bc3_random(V, t0)
        if z is None: continue
        chain = [t for t in sorted(larger) if t <= t0][::-1] if False else None
        # continue from t0 down to tc in the grid steps
        path = [t for t in sorted(TCS, reverse=True) if tc <= t < t0]
        zc = z.x
        for t in path:
            mt = g.Model(1.0, 10.0, V, t, 3, 1e-9, "cv"); Pt = N.Problem(mt); rt = ed.solve(1.0, 10.0, V, t, ht="cv")
            rng = np.random.default_rng(int(1000 * t))
            best = None
            for k in range(8):
                res = N.minimize_once(Pt, zc + 1e-3 * (k > 0) * rng.standard_normal(zc.size), maxiter=400)
                if accepted(Pt, res) is not None and (best is None or res.fun < best.fun):
                    best = res
                if best is not None and best.fun - rt["E0"] < 1.2e-7: break
            if best is None: break
            zc = best.x
        else:
            return best
    return None


if __name__ == "__main__":
    for V in VS:
        f = f"cederbaum_bc5_V{V}.npz"
        if not os.path.exists(f): continue
        data = dict(np.load(f)); changed = False
        for tc in TCS:
            if f"V{V}_tc{tc}_E" in data: continue
            t_start = time.time()
            r = ed.solve(1.0, 10.0, V, tc, ht="cv"); best3 = bc3_chain(V, tc)
            if best3 is None:
                print(f"V={V} t_c={tc}: continuation failed", flush=True); continue
            m3 = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P3 = N.Problem(m3)
            P5, z0 = embed(best3.x, V, tc); runs = []
            for nz in NOISES:
                for k in range(NSEED):
                    z = z0 + nz * np.random.default_rng(31 * k + int(1000 * nz) + int(100 * tc)).standard_normal(z0.size)
                    res = N.minimize_once(P5, z, maxiter=400); out = accepted(P5, res)
                    if out is not None: runs.append((res.fun, out, res))
            rr = np.random.default_rng(int(1000 * V + 7000 * tc) + 5); nrand_ok = 0
            for k in range(NRAND):
                res = N.minimize_once(P5, N.random_start(P5, rr), maxiter=400); out = accepted(P5, res)
                if out is not None: nrand_ok += 1; runs.append((res.fun, out, res))
            if not runs:
                print(f"V={V} t_c={tc}: no converged B_c=5 run", flush=True); continue
            runs.sort(key=lambda x: x[0]); E5, o5, _ = runs[0]
            key = f"V{V}_tc{tc}"
            data[key + "_E"] = np.array([r["E0"], r["nu_c"], best3.fun, N.recover(P3, best3.x)["nu"], E5, o5["nu"], o5["kkt"], len(runs),
                                         runs[-1][0] - runs[0][0], nrand_ok])
            data[key + "_poles"] = o5["poles"]; data[key + "_w"] = o5["weights"]; changed = True
            print(f"[{time.time()-t_start:4.0f}s] FIXED V={V} t_c={tc}: B3 dE={best3.fun-r['E0']:+.2e} | B5 dE={E5-r['E0']:+.2e} (B5-B3={E5-best3.fun:+.1e}) "
                  f"KKT={o5['kkt']:.0e} converged {len(runs)}/{2*NSEED+NRAND} main pole {o5['poles'][np.argmax(o5['weights'])]:.3f} w={o5['weights'].max():.4f}", flush=True)
        if changed: np.savez(f, **data)
    print("done", flush=True)
