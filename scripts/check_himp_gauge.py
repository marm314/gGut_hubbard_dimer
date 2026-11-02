"""Gauge shift of the multipliers -> H_imp of each fragment (loop formula, Fragment.solve): is Phi_I still an eigenstate, and which one (index in the sorted spectrum)?
lamc_I = lam_sum_I - lam_I (recover()), core with the Xi / alpha terms. GHOST_EPS=1e-7 python3 scripts/check_himp_gauge.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"
names = "cov"
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
rows = []
for V in (0.7071067811865476, 1.4142135623730951):
    data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
    for Bc in (3, 5):
        for tc in (0.4, 1.0, 2.0, 3.0, 5.0):
            z = data[f"tc{tc}_B{Bc}_z"]
            if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
            else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc, EPS); m = P.m
            E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
            U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
            rho, R, Dl = st["rho"], st["R"], st["Dl"]; alpha, Xi = m.core_extra(rho, R, Dl[0])
            def frag_info(mu):
                lam = [np.zeros((b, b)) for b in m.B]; row = 0
                for I in range(3):
                    for a, b in zip(*P.iu[I]): lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
                out = []
                for I in range(3):
                    H = himp(m, I, hyb[I], ls[I] - lam[I], alpha if I == 0 else 0.0, Xi if I == 0 else None); phi = st["phis"][I]
                    ev, U = np.linalg.eigh(H); e = phi @ H @ phi; resid = np.linalg.norm(H @ phi - e * phi)
                    ov = (U.T @ phi) ** 2; k = int(np.argmax(ov)); out.append((resid, k, ov[k], e - ev[0]))
                return out
            base = frag_info(mu0); idx = {I: [base[I][1]] for I in range(3)}; resmax = [base[I][0] for I in range(3)]
            rng = np.random.default_rng(4); dirs = list(np.eye(null.shape[1])) + ([] if null.shape[1] == 1 else list(rng.standard_normal((6, null.shape[1]))))
            for d in dirs:
                d = d / np.linalg.norm(d)
                for t in np.linspace(-300, 300, 61):
                    r = frag_info(mu0 + t * (null @ d))
                    for I in range(3): idx[I].append(r[I][1]); resmax[I] = max(resmax[I], r[I][0])
            print(f"V/g={V:.3f} tc={tc} B_c={Bc} ({null.shape[1]} dir): " + " | ".join(
                f"{names[I]}: eigenstate resid t=0 {base[I][0]:.1e} (max over shifts {resmax[I]:.1e}), index at t=0 {base[I][1]}, over shifts {min(idx[I])}..{max(idx[I])}, ground state reached: {'YES' if min(idx[I]) == 0 else 'no'}" for I in range(3)), flush=True)
