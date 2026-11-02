"""Impurity loop (gga_cederbaum.run, follow=False = lowest embedding state) started from the direct-minimization solution in different GAUGES of the multipliers.
B_c=3 (1 free direction). Scan t (mu = mu0 + t*null): Phi_c must be the non-degenerate ground state of H_imp,c and H_qp must obey aufbau; start the loop at the middle of that interval and, for contrast, at the min-norm gauge t=0.
GHOST_EPS=1e-7 python3 scripts/loop_at_gauge.py V tc [tc...]"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"
V = float(sys.argv[1]); TCS = [float(x) for x in sys.argv[2:]]; Bc = 3
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
data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
for tc in TCS:
    z = data[f"tc{tc}_B{Bc}_z"]; E0 = float(data[f"tc{tc}_E0"]); m = g.Model(1.0, 10.0, V, tc, Bc, EPS, "cv"); P = N.Problem(m)
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
    rho, R, Dl, q = st["rho"], st["R"], st["Dl"], st["q"]; alpha, Xi = m.core_extra(rho, R, Dl[0]); d = null[:, 0]
    def lam_of(mu):
        lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for a, b in zip(*P.iu[I]): lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
        return lam
    def status(t):
        lam = lam_of(mu0 + t * d); H = himp(m, 0, hyb[0], ls[0] - lam[0], alpha, Xi); ev, U = np.linalg.eigh(H); phi = st["phis"][0]
        ov0 = (U[:, 0] @ phi) ** 2; gap = ev[1] - ev[0]
        Hq = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, ph = np.linalg.eigh(Hq); occ = np.einsum("ik,ij,jk->k", ph, rho, ph); auf = e[occ > 0.5].max() < e[occ <= 0.5].min()
        return ov0, gap, auf
    ts = np.arange(-300, 300.5, 0.5); ok = [t for t in ts if (lambda r: r[0] > 0.999 and r[1] > 1e-3 and r[2])(status(t))]
    print(f"\n=== V/g={V:.3f} tc={tc} B_c=3 eps={EPS:g}: direct E-E_ED={E-E0:.2e}; gauge values with Phi_c = non-degenerate H_imp ground state AND aufbau: " + (f"t in [{min(ok):.1f}, {max(ok):.1f}] ({len(ok)} of {len(ts)} samples)" if ok else "NONE"), flush=True)
    cases = [("t=0 (min-norm gauge)", 0.0)] + ([(f"t={np.median(ok):.1f} (middle of ground-state+aufbau interval)", float(np.median(ok)))] if ok else [])
    for name, t in cases:
        lam = lam_of(mu0 + t * d); lamc = [ls[I] - lam[I] for I in range(3)]
        state0 = dict(R=R, lam=lam, nu=q[0]["nu"], K=q[0]["K"], rho=rho, hyb=hyb, lamc=lamc, Dl=Dl)
        ov0, gap, auf = status(t); print(f"  gauge {name}: overlap of Phi_c with H_imp ground state {ov0:.6f}, gap {gap:.2e}, aufbau {auf}", flush=True)
        for mix in (1.0, 0.5):
            sres = g.run(1.0, 10.0, V, tc, Bc, max_iter=300, mix=mix, tol=1e-10, follow=False, phi0=st["phis"], state0=state0, shift=EPS)
            print(f"    loop (lowest state) mix={mix}: converged={sres['converged']} it={sres['iters']} resid={sres['resid']:.1e} E_loop-E_direct={sres['E_var']-E:+.2e} nu_c={sres['nu']:.6f} (direct {q[0]['nu']:.6f}) "
                  f"main pole {sres['poles'][np.argmax(sres['weights'])]:.4f} w={sres['weights'].max():.4f} (direct {o['poles'][np.argmax(o['weights'])]:.4f})", flush=True)
