"""Muted (zero-weight) ghost poles of G^h_cc vs t_c/g for B_c=3,5 (B_o=B_v=1). Direct minimization (random starts + seeds from the
previous t_c / the B_c=3 solution for B_c=5), accepted if constraints<1e-9, KKT<1e-5 and ||[H_qp,rho]||<1e-5. Shift eps from env GHOST_EPS (default 1e-9; other values -> directory scan_ghost_tc_eps<eps>). Each converged point is saved at once.
Poles = occupied levels of H_qp (multipliers: min-norm least squares, as in recover()); the ghost-level positions are NOT unique (null
directions of the constraint Jacobian), so this is the min-norm choice. python3 scripts/scan_ghost_tc.py V [tc_list]
Output cederbaum_data/scan_ghost_tc/V{V}.npz, log scan_ghost_tc/V{V}.log"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, exact_cederbaum as ed, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
from compute_bc5_data import accepted as _acc
def accepted(P, res):          # constraints<1e-9, KKT_phi<1e-5 AND [H_qp,rho]<1e-5 (stationarity in X)
    o = _acc(P, res)
    return o if o is not None and o['comm'] < 1e-5 else None

EPS = float(os.environ.get("GHOST_EPS", "1e-9")); V = float(sys.argv[1]); TCS = [float(x) for x in os.environ['GHOST_TCS'].split(',')] if os.environ.get('GHOST_TCS') else [round(0.2 * i, 2) for i in range(1, 26)]
D = "cederbaum_data/scan_ghost_tc" + ("" if EPS == 1e-9 else f"_eps{EPS:g}"); os.makedirs(D, exist_ok=True); F = f"{D}/V{V}.npz"
data = dict(np.load(F)) if os.path.exists(F) else {}
def log(s): print(s, flush=True)
zprev = None
for tc in TCS:
    t0 = time.time(); r = ed.solve(1.0, 10.0, V, tc, ht="cv"); k = f"tc{tc}_"
    if k + "ed_poles" not in data:
        data[k + "ed_poles"] = r["poles"]; data[k + "ed_w"] = r["weights"]; data[k + "E0"] = r["E0"]
    m3 = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P3 = N.Problem(m3); rng = np.random.default_rng(int(1000 * V) + int(10 * tc)); best = None
    starts = ([zprev] if zprev is not None else []) + [None] * 40
    for i, z0 in enumerate(starts):
        res = N.minimize_once(P3, z0 if z0 is not None else N.random_start(P3, rng), maxiter=400)
        if accepted(P3, res) is not None and (best is None or res.fun < best.fun): best = res
        if best is not None and best.fun - r["E0"] < 120 * EPS: break
    if best is None: log(f"V={V} tc={tc}: B_c=3 FAILED ({time.time()-t0:.0f}s)"); continue
    zprev = best.x; o = N.recover(P3, best.x)
    data[k + "B3_z"] = best.x; data[k + "B3_poles"] = o["poles"]; data[k + "B3_w"] = o["weights"]; data[k + "B3_dE"] = best.fun - r["E0"]; data[k + "B3_kkt"] = o["kkt"]
    log(f"V={V} tc={tc}: B_c=3 dE={best.fun-r['E0']:.2e} kkt={o['kkt']:.1e} poles={np.round(o['poles'],3)} w={np.array2string(o['weights'],precision=1)} ({time.time()-t0:.0f}s)")
    P5, z0 = embed(best.x, V, tc, EPS); b5 = None
    for nz in (0.03, 0.1):
        for kk in range(4):
            res = N.minimize_once(P5, z0 + nz * np.random.default_rng(31 * kk + int(1000 * nz)).standard_normal(z0.size), maxiter=400)
            if accepted(P5, res) is not None and (b5 is None or res.fun < b5.fun): b5 = res
    if b5 is None: log(f"V={V} tc={tc}: B_c=5 FAILED"); np.savez(F, **data); continue
    o5 = N.recover(P5, b5.x)
    data[k + "B5_z"] = b5.x; data[k + "B5_poles"] = o5["poles"]; data[k + "B5_w"] = o5["weights"]; data[k + "B5_dE"] = b5.fun - r["E0"]; data[k + "B5_kkt"] = o5["kkt"]
    np.savez(F, **data)
    log(f"V={V} tc={tc}: B_c=5 dE={b5.fun-r['E0']:.2e} kkt={o5['kkt']:.1e} poles={np.round(o5['poles'],3)} w={np.array2string(o5['weights'],precision=1)} ({time.time()-t0:.0f}s)")
