"""B_c=3: where do the weightless ghost levels sit when Phi_c is the non-degenerate ground state of H_imp,c and H_qp obeys aufbau? vs ED satellites.
GHOST_EPS=1e-7 python3 scripts/ghost_in_gs_window.py"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"
def himp(m, I, hyb, lamc, alpha, Xi):
    f = m.frag[I]; B = f.B; H = f.Hloc.copy()
    for a in range(B):
        H += hyb[a] * f.hyb_op[a]
        for b in range(B): H += lamc[a, b] * f.dd[a][b]
    if Xi is not None:
        H += (alpha + 2.0 * np.trace(Xi)) * f.nc
        for gg in range(B):
            for d in range(B): H -= Xi[gg, d] * f.nc_dd[gg][d]
    return H
for V in (0.7071067811865476, 1.4142135623730951):
    data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
    for tc in (0.4, 1.0, 2.0, 3.0, 5.0):
        z = data[f"tc{tc}_B3_z"]; m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
        E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
        U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
        rho, R, Dl = st["rho"], st["R"], st["Dl"]; alpha, Xi = m.core_extra(rho, R, Dl[0]); d = null[:, 0]
        def lam_of(mu):
            lam = [np.zeros((b, b)) for b in m.B]; row = 0
            for I in range(3):
                for a, b in zip(*P.iu[I]): lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
            return lam
        rows = []
        for t in np.arange(-300, 300.5, 0.5):
            lam = lam_of(mu0 + t * d); H = himp(m, 0, hyb[0], ls[0] - lam[0], alpha, Xi); ev, U = np.linalg.eigh(H); phi = st["phis"][0]
            Hq = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, ph = np.linalg.eigh(Hq); occ = np.einsum("ik,ij,jk->k", ph, rho, ph); wc = (R[0] @ ph[m.sl[0], :]) ** 2
            if (U[:, 0] @ phi) ** 2 > 0.999 and ev[1] - ev[0] > 1e-3 and e[occ > 0.5].max() < e[occ <= 0.5].min():
                isg = (occ > 0.5) & ((ph[m.sl[0], :] ** 2).sum(0) > 0.9) & (wc < 1e-2); rows.append((t, np.sort(e[isg])[0] if isg.any() else np.nan, e[np.argmax(wc)]))
        r = ed = None
        import exact_cederbaum as exm
        r = exm.solve(1.0, 10.0, V, tc, ht="cv"); pm = r["poles"][r["weights"] > 1e-3]
        if rows:
            ghl = [x[1] for x in rows]; print(f"V/g={V:.3f} tc={tc}: window t in [{rows[0][0]:.1f},{rows[-1][0]:.1f}]; ghost level in window {min(ghl):.2f} .. {max(ghl):.2f}; QP {rows[0][2]:.2f}; ED poles (w>1e-3) {np.round(pm,2)} weights {np.round(r['weights'][r['weights']>1e-3],3)}")
        else: print(f"V/g={V:.3f} tc={tc}: NO window; ED poles {np.round(pm,2)}")
