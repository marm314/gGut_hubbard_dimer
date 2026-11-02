"""Stationarity vs the regularizing shift eps: re-minimize the stored eps=1e-9 solution (V/g=sqrt2, t_c/g=1.0, B_c=3) at several eps; E-E_ED, projected gradient (full KKT residual), ||[H_qp,rho]||. PYTHONPATH=scripts python3 scripts/eps_dependence.py"""
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g, not_imp_gga_cederbaum as N
V, tc = 1.4142135623730951, 1.0
data = np.load(f"cederbaum_data/scan_ghost_tc/V{V}.npz"); z0 = data[f"tc{tc}_B3_z"]; E0 = float(data[f"tc{tc}_E0"])
for eps in (1e-9, 1e-7, 1e-5, 1e-12):
    m = g.Model(1.0, 10.0, V, tc, 3, eps, "cv"); P = N.Problem(m); z = z0.copy(); cons = {"type": "eq", "fun": P.constraints, "jac": P.cons_jac}
    for _ in range(3):
        r = minimize(lambda x: P.energy_and_grad(x)[0], z, jac=lambda x: P.energy_and_grad(x)[1], constraints=[cons], method="SLSQP", options=dict(maxiter=1500, ftol=1e-15)); z = r.x
    E, gz, *_ = P.energy_and_grad(z); J = P.cons_jac(z); mu, *_ = np.linalg.lstsq(J.T, -gz, rcond=None); o = N.recover(P, z); sv = np.linalg.svd(J[:, :P.nx], compute_uv=False)
    print(f"eps={eps:.0e}: dE={E-E0:+.2e} |cons|={np.abs(P.constraints(z)).max():.0e} projected grad={np.linalg.norm(J.T@mu+gz):.1e} comm={o['comm']:.1e} smallest sv(J_phi)={sv[-1]:.1e}  [{r.message[:28]}]", flush=True)
