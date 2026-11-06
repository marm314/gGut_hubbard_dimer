"""Cederbaum gGut (B_c=3 by default) at a FIXED core |R_c|: direct minimization (not_imp_gga_cederbaum, SLSQP) with the extra equality
constraint |R_c|^2 = (s |R_c^0|)^2, continuation in s from the unconstrained solution (s=1) downward (and optionally upward).
Question: does lowering |R_c| wake the sleeping ghosts (nonzero weight) and where do their poles sit relative to the ED satellites?
Multipliers of the density constraints from the joint least squares J_phi^T mu + nu grad|R_c|^2 = -grad_phi E (the R-constraint does not
depend on X, so [H_qp, rho]=0 is unchanged). Every converged point is saved at once (cederbaum_data/fixedR/V<V>_tc<tc>_B<Bc>_eps<eps>/s_<s>.npz)
and logged: s, E-E_ED, constraints, KKT, comm, poles, weights, ED poles/weights.
python3 fixedR_cederbaum.py V tc Bc eps s_min ds [seed_npz_key]"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g, exact_cederbaum as ed, not_imp_gga_cederbaum as N

V, tc, Bc, EPS, smin, ds = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
r = ed.solve(1.0, 10.0, V, tc, ht="cv"); E0 = r["E0"]
m = g.Model(1.0, 10.0, V, tc, Bc, EPS, "cv"); P = N.Problem(m)
src = f"../cederbaum_data/scan_ghost_tc_eps{EPS:g}/V{V}.npz"
z0 = np.load(src)[f"tc{tc}_B{Bc}_z"]
out = f"../cederbaum_data/fixedR/V{V:.4f}_tc{tc}_B{Bc}_eps{EPS:g}"; os.makedirs(out, exist_ok=True)
Rc2 = lambda z: float(P.state(z)["R"][0] @ P.state(z)["R"][0])
def Rc2_grad(z, h=1e-6):
    return np.array([(Rc2(z + h * e) - Rc2(z - h * e)) / (2 * h) for e in np.eye(len(z))[:P.nx]] ).tolist()
def gradfull(z):
    gg = np.zeros(len(z)); gg[:P.nx] = Rc2_grad(z); return gg
def solve(z, target):
    cons = [{"type": "eq", "fun": P.constraints, "jac": P.cons_jac},
            {"type": "eq", "fun": lambda x: Rc2(x) - target, "jac": gradfull}]
    return minimize(lambda x: P.energy_and_grad(x)[0], z, jac=lambda x: P.energy_and_grad(x)[1], constraints=cons, method="SLSQP", options=dict(maxiter=600, ftol=1e-13))
def analyze(z):
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]
    A = np.column_stack([J.T, gradfull(z)[:P.nx]]); sol, *_ = np.linalg.lstsq(A, -gz[:P.nx], rcond=None)
    kkt = np.linalg.norm(A @ sol + gz[:P.nx]); mu = sol[:-1]
    lam = [np.zeros((b, b)) for b in m.B]; row = 0
    for I in range(3):
        for a, b in zip(*P.iu[I]):
            lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
    H = m.build_Hqp(st["R"], lam, Tb, u, w, st["q"][0]["nu"]); e, ph = np.linalg.eigh(H)
    wts = np.array([(st["R"][0] @ ph[m.sl[0], k]) ** 2 for k in range(m.Nocc)])
    return dict(E=E, nu=st["q"][0]["nu"], kkt=kkt, comm=np.linalg.norm(H @ st["rho"] - st["rho"] @ H), poles=e[:m.Nocc], w=wts, nuR=sol[-1], Rc2=Rc2(z))
def log(s): print(s, flush=True)
sat = r["weights"] > 1e-3
log(f"V={V} tc={tc} B_c={Bc} eps={EPS:g}: ED E0={E0:.9f}; ED poles(w>1e-3) {np.round(r['poles'][sat],3)} w {np.round(r['weights'][sat],3)}")
a = analyze(z0); R2_0 = a["Rc2"]
log(f"s=1.0000: E-E_ED={a['E']-E0:.3e} |R_c|^2={R2_0:.6f} kkt={a['kkt']:.1e} comm={a['comm']:.1e} poles {np.round(a['poles'],3)} w {np.array2string(a['w'],precision=4)}")
np.savez(f"{out}/s_1.0000.npz", z=z0, **{k: v for k, v in a.items()})
def walk(direction, limit):
    s, z, h = 1.0, z0, ds
    while (limit - s) * direction > 1e-9:
        sn = round(s + direction * min(h, abs(limit - s)), 6); t0 = time.time()
        res = solve(z, sn * sn * R2_0); viol = max(np.abs(P.constraints(res.x)).max(), abs(Rc2(res.x) - sn * sn * R2_0)); a = analyze(res.x)
        ok = bool(res.success) and viol < 1e-7 and a["comm"] < 1e-4
        log(f"s={sn:.4f} (h={abs(sn-s):.3f}) E-E_ED={res.fun-E0:.3e} viol={viol:.1e} succ={res.success} kkt={a['kkt']:.1e} comm={a['comm']:.1e} nit={res.nit} {time.time()-t0:.0f}s -> {'ok' if ok else 'REJECT'}")
        if ok:
            log(f"    poles {np.round(a['poles'],3)} w {np.array2string(a['w'],precision=4)} nu_c={a['nu']:.6f}")
            np.savez(f"{out}/s_{sn:.4f}.npz", z=res.x, **{k: v for k, v in a.items()}); s, z, h = sn, res.x, min(1.5 * h, ds)
        else:
            h /= 2
            if h < 1e-3: log(f"walk stopped at s={s:.4f}"); return
walk(-1, smin)
