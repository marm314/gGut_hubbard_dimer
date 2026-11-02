"""Observables at gauge-shifted multipliers (eps=1e-7 solutions). For each shift mu0 + t*null@d:
 (a) E, nu_c, double occupancies <n_up n_dn> of c,o,v from Phi_I and rho (z only: cannot change);
 (b) rho' rebuilt as the projector on the rho-occupied eigenvectors of the shifted H_qp, E' = Model.energy(rho', ...), |rho'-rho|, QP pole/weight, ghost levels.
GHOST_EPS=1e-7 python3 scripts/check_gauge_observables.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"
def proj(f, M): return M[f.sec][:, f.sec].toarray()
for V, tc, Bc in [(0.7071067811865476, 1.0, 3), (0.7071067811865476, 3.0, 5), (1.4142135623730951, 1.0, 3), (1.4142135623730951, 3.0, 5)]:
    data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz"); z = data[f"tc{tc}_B{Bc}_z"]
    if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
    else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc, EPS); m = P.m
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
    docc = [st["phis"][I] @ proj(m.frag[I], m.frag[I].ops["n_pu"] @ m.frag[I].ops["n_pd"]) @ st["phis"][I] for I in range(3)]
    rho, R, Dl, q = st["rho"], st["R"], st["Dl"], st["q"]
    print(f"\nV/g={V:.4f} t_c/g={tc} B_c={Bc} (eps={EPS:g}; {null.shape[1]} free directions)  E={E:.10f}  nu_c={q[0]['nu']:.10f}  <n_up n_dn> c,o,v = {docc[0]:.8f} {docc[1]:.8f} {docc[2]:.8f}   [from z: identical for every shift]")
    print("   shift t |  E' from rho'(H_qp)  |rho'-rho| | QP pole    QP weight  | ghost levels")
    d = np.random.default_rng(3).standard_normal(null.shape[1]); d /= np.linalg.norm(d)
    for t in (-100, -20, 0, 20, 100):
        mu = mu0 + t * (null @ d); lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for a, b in zip(*P.iu[I]): lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
        H = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, phi = np.linalg.eigh(H); occ = np.einsum("ik,ij,jk->k", phi, rho, phi)
        pocc = phi[:, occ > 0.5]; rho2 = pocc @ pocc.T
        E2 = m.energy(rho2, R, Dl[0], q[0]["nu"], q[0]["K"], sum(i["Eloc"] for i in q))
        wc = (R[0] @ phi[m.sl[0], :]) ** 2; k = np.argmax(wc); isg = (occ > 0.5) & ((phi[m.sl[0], :] ** 2).sum(0) > 0.9) & (wc < 1e-2)
        print(f"   {t:5d}  | {E2:.10f}   {np.linalg.norm(rho2-rho):.1e} | {e[k]:.7f} {wc[k]:.7f} | {np.round(np.sort(e[isg]),2)}")
