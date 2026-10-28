"""At a direct-minimization solution: build H_qp (multipliers lam from recover), diagonalize, fill the N_occ lowest levels,
and compare with the direct rho (is rho a fixed point of the inner loop rho -> H_qp -> rho?).
Usage: python3 check_direct_rho_fixed.py Bc V shift"""
import os, sys, numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N

Bc, V, shift = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tc, NS = 0.5, int(os.environ.get("NS", 5))
m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng = np.random.default_rng(0)
best = None
for k in range(NS):
    res = N.minimize_once(P, N.random_start(P, rng))
    if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
        best = res
out = N.recover(P, best.x); q0 = out["st"]["q"][0]
R, lam, rho, Dl, nu, K = out["R"], out["lam"], out["rho"], out["Dl"], q0["nu"], q0["K"]
u, w = m.uw(rho, R); Tb = m.theta_bar(Dl[0], nu, K)
H = m.build_Hqp(R, lam, Tb, u, w, nu)
e, phi, rho2 = m.qp_state(H)
print(f"Bc={Bc} V={V} shift={shift:g}: E={best.fun:.10f} KKT={out['kkt']:.1e} cons={np.abs(P.constraints(best.x)).max():.1e}")
print(f"  ||[H,rho_direct]||={np.linalg.norm(H@rho-rho@H):.2e}   ||rho_filled(H) - rho_direct||={np.linalg.norm(rho2-rho):.2e}")
print(f"  eigenvalues of H: {np.round(e,5)}  (N_occ={m.Nocc}; gap eps_Nocc+1 - eps_Nocc = {e[m.Nocc]-e[m.Nocc-1]:.3e})")
occ = np.diag(phi.T @ rho @ phi)
print(f"  occupation of H eigenstates in rho_direct: {np.round(occ,6)}")
