"""Numerical check of Sec. 3 of doc/excited_phi_cederbaum.tex at V/g = 1/sqrt2, sqrt2 (t_c=0.5, B_c=3, shift 1e-9):
is Phi_I from the direct minimization an eigenstate of H_imp^I, which one (index), energies, bath levels of lambda^c_c,
eigenvalues of ^c Delta. python3 excited_check_golze.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe

tc, Bc, shift = 0.5, 3, 1e-9
for V in pe.VS:
    m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng = np.random.default_rng(0); best = None
    for k in range(5):
        res = N.minimize_once(P, N.random_start(P, rng))
        if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun): best = res
    out = N.recover(P, best.x); st = out["st"]
    R, rho, Dl, hyb, lamc = out["R"], out["rho"], out["Dl"], out["hyb"], out["lamc"]
    phis = st["phis"]
    alpha, Xi = m.core_extra(rho, R, Dl[0])
    print(f"\nV={V:.4f}: direct E={best.fun:.8f} kkt={out['kkt']:.1e}")
    for I, name in enumerate("cov"):
        args = (hyb[I], lamc[I]) + ((alpha, Xi) if I == 0 else ())
        f_ref = m.frag[I].solve(*args, ref=phis[I]); f_low = m.frag[I].solve(*args)
        print(f"  Phi_{name}: overlap^2 with the selected eigenstate = {(f_ref['gs'] @ phis[I])**2:.6f}; index {f_ref['index']}; "
              f"<Phi|H|Phi> = {f_ref['E']:.4f}; lowest eigenvalue {f_low['E']:.4f}")
    print("  bath levels of lambda^c_c:", np.round(np.linalg.eigvalsh(lamc[0]), 2), " eig(^c Delta):", np.linalg.eigvalsh(Dl[0]))

# ---- one loop step from the direct-minimization state (shift 1e-9) with lowest-state / max-overlap selection,
#      and the spacing of the embedding levels around the followed core state
print("\n== one step from the direct-minimization state, shift 1e-9")
for V in pe.VS:
    m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng = np.random.default_rng(0); best = None
    for k in range(5):
        res = N.minimize_once(P, N.random_start(P, rng))
        if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun): best = res
    out = N.recover(P, best.x); st = out["st"]; q0 = st["q"][0]
    state0 = dict(R=out["R"], lam=out["lam"], nu=q0["nu"], K=q0["K"], rho=out["rho"], hyb=out["hyb"], lamc=out["lamc"], Dl=out["Dl"])
    for follow in (False, True):
        s = g.run(1.0, 10.0, V, tc, Bc, max_iter=1, mix=1.0, tol=1e-10, follow=follow, phi0=st["phis"], state0=state0, shift=shift)
        print(f"  V={V:.4f} follow={follow!s:5}: residual after 1 step {s['resid']:.2e}")
    fr = m.frag[0]; hyb, lamc = out["hyb"][0], out["lamc"][0]
    alpha, Xi = m.core_extra(out["rho"], out["R"], out["Dl"][0])
    H = fr.Hloc.copy()
    for a in range(Bc):
        H += hyb[a] * fr.hyb_op[a]
        for b in range(Bc):
            H += lamc[a, b] * fr.dd[a][b]
    H += (alpha + 2.0 * np.trace(Xi)) * fr.nc
    for gg in range(Bc):
        for d in range(Bc):
            H -= Xi[gg, d] * fr.nc_dd[gg][d]
    E, U = np.linalg.eigh(H); k = int(np.argmax((U.T @ st["phis"][0]) ** 2))
    sp = np.diff(E)
    print(f"  V={V:.4f}: core state index {k}; spacing to the neighbours {sp[k-1]:.2e} (below), {sp[k]:.2e} (above); smallest spacing among the lowest {k+4} levels {sp[:k+3].min():.2e}")
