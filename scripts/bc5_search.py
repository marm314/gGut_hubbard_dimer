"""Sequential B_c=5 search by direct minimization, one run at a time, one log line per run.
Per (V, t_c): (1) B_c=3 solution (best of a few random starts); (2) B_c=5 starts seeded from it with the two extra core
ghosts asleep, perturbed by NOISES; (3) purely random B_c=5 starts. Everything is compared with the B_c=3 energy and ED.
Run with:  python3 scripts/bc5_search.py [V t_c]   (default: the three t_c=0.1 cases)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
from bc5_cederbaum import embed

NOISES = [0.03, 0.1, 0.2, 0.3]
NSEED = 6
NRAND = 24


def line(tag, P, res, e_ref, e_ed, t0):
    cons = np.abs(P.constraints(res.x)).max()
    out = N.recover(P, res.x)
    ok = cons < 1e-9 and out["kkt"] < 1e-5
    print(f"  [{time.time()-t0:6.0f}s] {tag:16s} E-E_ED={res.fun-e_ed:+.3e} (vs B3 {res.fun-e_ref:+.2e}) cons={cons:.0e} KKT={out['kkt']:.0e} "
          f"it={res.nit:3d} {'OK ' if ok else 'bad'} Rc2={out['R'][0]@out['R'][0]:.4f} main pole {out['poles'][np.argmax(out['weights'])]:.3f} "
          f"(w={out['weights'].max():.4f}) other weights {np.round(np.sort(out['weights'])[:-1][::-1],5)}", flush=True)
    return ok, out


def case(V, tc):
    t0 = time.time()
    r = ed.solve(1.0, 10.0, V, tc, ht="cv")
    print(f"\n=== V={V} t_c={tc}: ED E0={r['E0']:.8f} nu_c={r['nu_c']:.6f}", flush=True)
    m3 = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P3 = N.Problem(m3); rng = np.random.default_rng(int(1000 * V + 7000 * tc))
    best3 = None
    for k in range(20):
        res = N.minimize_once(P3, N.random_start(P3, rng), maxiter=400)
        if np.abs(P3.constraints(res.x)).max() < 1e-9 and N.recover(P3, res.x)["kkt"] < 1e-5 and (best3 is None or res.fun < best3.fun):
            best3 = res
        if best3 is not None and best3.fun - r["E0"] < 1.2e-7:
            break
    if best3 is None:
        print("  no B_c=3 solution found"); return
    e3 = best3.fun
    print(f"  [{time.time()-t0:6.0f}s] B_c=3 solution: E-E_ED={e3-r['E0']:+.3e}", flush=True)
    P5, z0 = embed(best3.x, V, tc)
    found = []
    for nz in NOISES:
        for k in range(NSEED):
            z = z0 + nz * np.random.default_rng(10 * k + int(100 * nz)).standard_normal(z0.size)
            res = N.minimize_once(P5, z, maxiter=300)
            ok, out = line(f"seeded noise {nz:g}", P5, res, e3, r["E0"], t0)
            if ok: found.append(res.fun)
    rng = np.random.default_rng(7)
    for k in range(NRAND):
        res = N.minimize_once(P5, N.random_start(P5, rng), maxiter=400)
        ok, out = line(f"random #{k}", P5, res, e3, r["E0"], t0)
        if ok: found.append(res.fun)
    if found:
        print(f"  SUMMARY V={V} t_c={tc}: {len(found)} converged B_c=5 runs; lowest E-E_B3 = {min(found)-e3:+.3e}; "
              f"highest = {max(found)-e3:+.3e}", flush=True)


if __name__ == "__main__":
    cases = [(float(sys.argv[1]), float(sys.argv[2]))] if len(sys.argv) > 2 else [(1.0, 0.1), (0.5, 0.1), (2.0, 0.1)]
    for V, tc in cases:
        case(V, tc)
