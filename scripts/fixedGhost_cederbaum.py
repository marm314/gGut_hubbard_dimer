"""Give weight to the (pure, filled) ghost of B_c=3 and place its pole on an ED satellite.
Constraint: c_g = |R_c . u_g| = t (pole weight ~ t^2; the linear amplitude avoids the degenerate gradient of the squared weight at 0), u_g = natural orbital of Delta_c with the largest occupation (the pure filled ghost, n=1);
direct minimization (SLSQP, not_imp_gga_cederbaum.Problem) with continuation in t from the unconstrained solution (t~0) upward.
Multipliers: joint least squares J_phi^T mu + nu grad w_g = -grad_phi E (w_g does not depend on X, so [H_qp,rho]=0 as before).
Then the gauge shift a (null direction of J_phi^T, H_qp(a) = H_qp(0) + a D exactly) moves the ghost-related occupied level; the scan over a lists
poles/weights of G_cc (every occupied level) and the a that puts a level at the target ED satellite (default the heaviest satellite below the main pole).
Each converged point is saved at once. python3 fixedGhost_cederbaum.py V tc eps t_list(comma) [target_pole]"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g, exact_cederbaum as ed, not_imp_gga_cederbaum as N

V, tc, EPS = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); TS = [float(x) for x in sys.argv[4].split(",")]
r = ed.solve(1.0, 10.0, V, tc, ht="cv"); E0 = r["E0"]; sat = r["weights"] > 1e-3
below = [(p, w) for p, w in zip(r["poles"][sat], r["weights"][sat]) if p < r["poles"][sat].max() - 0.5 and p > -13]
target = float(sys.argv[5]) if len(sys.argv) > 5 else max(below, key=lambda x: x[1])[0]
m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
z0 = np.load(f"../cederbaum_data/scan_ghost_tc_eps{EPS:g}/V{V}.npz")[f"tc{tc}_B3_z"]
out = f"../cederbaum_data/fixedGhost/V{V:.4f}_tc{tc}_eps{EPS:g}"; os.makedirs(out, exist_ok=True)


def ug(z):
    st = P.state(z); n, U = np.linalg.eigh(st["Dl"][0]); u = U[:, -1]; return st, n, u


def wg(z):
    st, n, u = ug(z); return float(abs(st["R"][0] @ u))


def wg_grad(z, h=1e-6):
    gg = np.zeros(len(z)); gg[:P.nx] = [(wg(z + h * e) - wg(z - h * e)) / (2 * h) for e in np.eye(len(z))[:P.nx]]; return gg


def solve(z, t):
    cons = [{"type": "eq", "fun": P.constraints, "jac": P.cons_jac}, {"type": "eq", "fun": lambda x: wg(x) - t, "jac": wg_grad}]
    return minimize(lambda x: P.energy_and_grad(x)[0], z, jac=lambda x: P.energy_and_grad(x)[1], constraints=cons, method="SLSQP",
                    options=dict(maxiter=2500, ftol=1e-13))


def analyze(z):
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; G = wg_grad(z)[:P.nx]
    A = np.column_stack([J.T, G]); sol, *_ = np.linalg.lstsq(A, -gz[:P.nx], rcond=None); kkt = np.linalg.norm(A @ sol + gz[:P.nx]); mu0 = sol[:-1]
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]

    def Hq(a):
        mu = mu0 + null @ a; lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for i, j in zip(*P.iu[I]): lam[I][i, j] = lam[I][j, i] = (-mu[row] / 2.0 if i == j else -mu[row] / 4.0); row += 1
        return m.build_Hqp(st["R"], lam, Tb, u, w, st["q"][0]["nu"])

    def spec(a):
        H = Hq(a); e, ph = np.linalg.eigh(H); occ = np.einsum("ik,ij,jk->k", ph, st["rho"], ph); wc = (st["R"][0] @ ph[m.sl[0], :]) ** 2
        idx = np.flatnonzero(occ > 0.5); return e[idx], wc[idx], H

    H0 = Hq(np.zeros(null.shape[1]))
    return dict(E=E, kkt=kkt, st=st, null=null, rank=rank, spec=spec, comm=np.linalg.norm(H0 @ st["rho"] - st["rho"] @ H0))


def log(s): print(s, flush=True)


log(f"V={V:.4f} tc={tc} eps={EPS:g}: ED poles(w>1e-3) {np.round(r['poles'][sat],3)} w {np.round(r['weights'][sat],3)}; target pole {target:.3f}")
z = z0
for t in TS:
    t0 = time.time(); res = solve(z, t); viol = max(np.abs(P.constraints(res.x)).max(), abs(wg(res.x) - t)); a = analyze(res.x)
    st, n, u = ug(res.x); ok = viol < 1e-6 and a["kkt"] < 1e-3 and a["comm"] < 1e-3
    log(f"c={t:.2e} (w={t*t:.1e}): E-E_ED={res.fun-E0:.3e} viol={viol:.1e} succ={res.success} nit={res.nit} kkt={a['kkt']:.1e} comm={a['comm']:.1e} "
        f"nat.occ Delta_c {np.array2string(n,precision=5)} nu_c={st['q'][0]['nu']:.6f} {time.time()-t0:.0f}s -> {'ok' if ok else 'REJECT'}")
    if not ok: break
    z = res.x; na = a["null"].shape[1]
    e0, w0, _ = a["spec"](np.zeros(na)); log(f"    a=0 occupied poles {np.round(e0,3)} w {np.array2string(w0,precision=4)} (rank {a['rank']}, {na} gauge dir)")
    if na == 1:
        best = None
        for av in np.linspace(-30, 30, 6001):
            e, wv, _ = a["spec"](np.array([av]))
            if len(e) == len(e0):
                d = np.abs(e - target); k = int(np.argmin(d))
                if best is None or d[k] < best[0]: best = (d[k], av, e.copy(), wv.copy())
        if best: log(f"    gauge a={best[1]:.3f}: poles {np.round(best[2],3)} w {np.array2string(best[3],precision=4)} (nearest to target {target:.3f}: miss {best[0]:.3f})")
    np.savez(f"{out}/t_{t:.2e}.npz", z=res.x, E=res.fun, w=t)
