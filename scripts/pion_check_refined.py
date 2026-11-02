"""At the dV=2 confirmed points: refine by direct minimization (seed at confirmed R,lambda), take the refined (R,lambda) from analyse(),
and compute P_ion both ways at THAT point: Phi-matrix form and impurity ground state. Also the impurity value at the confirmed (R,lambda).
Run from repo root."""
import sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni, gga_dimer as gg, make_qp_orbital_videos as q
t = 1.0
def imp_pion(R, lam, U, dV=2.0):
    Neff = len(R[0]); ops = [gg.build_impurity_ops(Neff, U, U/2+dV), gg.build_impurity_ops(Neff, U, U/2-dV)]
    Hq, rho = gg.qp_state(R, lam, t); bl = [rho[:Neff,:Neff], rho[Neff:,Neff:]]
    V = [gg.hybridization_V(rho, R, t, 1e-9, i) for i in range(2)]
    lc = [gg.embedding_lambda_c(bl[i], lam[i], V[i], R[i], 1e-9) for i in range(2)]
    imp = [gg.impurity_solve(V[i], lc[i], ops[i]) for i in range(2)]
    return imp[0][3] + imp[1][3]
for U in (7.0, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.0):
    R, lam = q.load_R_lambda(f"scan_ng2_dV2_confirmed/U_{U:.1f}")
    m = ni.NotImpDimer(2, U, deltaV=2.0)
    z = m.minimize(m.start_from_R_lambda(R, lam)).x
    a = m.analyse(z)
    Rr = [a["R0"], a["R1"]]; lr = [a["lam0"], a["lam1"]]
    pphi = sum((x @ m.Ndd @ x)/(x @ x) for x in m.unpack(z)[:2])
    print(f"U={U:.1f} confirmed:imp={imp_pion(R,lam,U):.7f} | refined: phi={pphi:.7f} imp={imp_pion(Rr,lr,U):.7f} E={a['E_var']:.8f} kkt_imp={a['kkt_imp']:.1e} dR={max(np.abs(np.array(R)-np.array(Rr)).max(),0):.1e}", flush=True)
