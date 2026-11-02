"""Retry failed B_c=5 points of scan_ghost_tc.py: more noisy seeds around the embedded B_c=3 solution plus seeds from neighbouring converged B_c=5 solutions."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, exact_cederbaum as ed, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
from compute_bc5_data import accepted as _acc
def accepted(P, res):
    o = _acc(P, res)
    return o if o is not None and o['comm'] < 1e-5 else None
EPS = float(os.environ.get("GHOST_EPS", "1e-9")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"; V = float(sys.argv[1]); F = f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz"; data = dict(np.load(F))
for tc in [round(0.2 * i, 2) for i in range(1, 26)]:
    k = f"tc{tc}_"
    if k + "B3_z" not in data or k + "B5_z" in data: continue
    E0 = data[k + "E0"]; P5, z0 = embed(data[k + "B3_z"], V, tc, EPS); seeds = []
    for dt in (-0.2, 0.2, -0.4, 0.4):
        kk = f"tc{round(tc+dt,2)}_B5_z"
        if kk in data: seeds.append(data[kk])
    b5 = None; rng = np.random.default_rng(5)
    for nz in (0.01, 0.03, 0.1, 0.2, 0.3):
        for _ in range(6):
            for zs in [z0] + seeds:
                res = N.minimize_once(P5, zs + nz * rng.standard_normal(zs.size), maxiter=600)
                if accepted(P5, res) is not None and (b5 is None or res.fun < b5.fun): b5 = res
        if b5 is not None: break
    if b5 is None: print(f"tc={tc}: still FAILED", flush=True); continue
    o5 = N.recover(P5, b5.x)
    data[k + "B5_z"] = b5.x; data[k + "B5_poles"] = o5["poles"]; data[k + "B5_w"] = o5["weights"]; data[k + "B5_dE"] = b5.fun - E0; data[k + "B5_kkt"] = o5["kkt"]
    np.savez(F, **data); print(f"tc={tc}: B_c=5 dE={b5.fun-E0:.2e} kkt={o5['kkt']:.1e} poles={np.round(o5['poles'],3)}", flush=True)
