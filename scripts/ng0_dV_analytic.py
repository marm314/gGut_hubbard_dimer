"""Ng=0 half-filled dimer with site splitting dV: reduction to two variables and closed-form U_c.
Site-exchange x particle-hole symmetry gives phi_1=(s,eps,delta) from phi_0=(s,delta,eps) in the basis (S,D,E) of the embedding sector, hence Z0=Z1. With
c=delta^2+eps^2, m=delta^2-eps^2 (= polarization = n_favored - 1), r=sqrt(c^2-m^2):
  E(c,m) = -U(1-c) - 2 dV m - 4t (1-c)(c+r)/sqrt(1-m^2),   P_ion=c,   Z = 2(1-c)(c+r)/(1-m^2),   qp levels +-tZ/sqrt(1-m^2) with weight 2Z each.
Stationarity:  U sqrt(1-m^2) r = 4t (c+r)(1-c-r),   dV (1-m^2)^(3/2) r = 2t (1-c) m (1-c^2-r c).
Z -> 0 at c,m -> 0 with q=r/c: U_c = 4t(1+q)/q, dV = 2t sqrt(1-q^2)/q  =>  U_c = 4t + 2 sqrt(4t^2+dV^2).
Checks against scan_ng0/ (numerical NotImpDimer(Ng=0)) for E, Z, P_ion, qp level, and U_c. Run from the repo root."""
import numpy as np
from scipy.optimize import minimize
t = 1.0
def E(x, U, dV):
    """energy in the amplitudes (delta, eps) of the embedding state (s^2 = 1 - delta^2 - eps^2); same function as E(c,m) with c=delta^2+eps^2, m=delta^2-eps^2"""
    d, e = x; w = 1 - d * d - e * e; return -U * w - 2 * dV * (d * d - e * e) - 2 * t * w * (d + e) ** 2 / np.sqrt((d * d + w / 2) * (e * e + w / 2) + 1e-300)
def solve(U, dV, x0):
    best = None
    for s0 in (x0, (0.5, 0.5), (0.6, 0.3), (0.3, 0.1), (0.1, 0.05), (0.02, 0.01), (0.7, 0.1)):
        r = minimize(E, s0, args=(U, dV), method="SLSQP", bounds=[(0, 1), (0, 1)], constraints=[{"type": "ineq", "fun": lambda x: 1 - x[0] ** 2 - x[1] ** 2}], options=dict(ftol=1e-15, maxiter=500))
        if best is None or r.fun < best.fun - 1e-13: best = r
    return best
def quantities(c, m, U, dV):
    r = np.sqrt(max(c * c - m * m, 0)); Z = 2 * (1 - c) * (c + r) / (1 - m * m); return Z, c, 1 + m, t * Z / np.sqrt(1 - m * m)
if __name__ == "__main__":
    worst = dict(E=0, Z=0, P=0, eps=0)
    for dV in (0.0, 0.4, 1.0, 2.0, 3.0, 5.0, 10.0):
        tab = np.loadtxt(f"scan_ng0/table_dV{dV:.1f}.txt"); x = (0.5, 0.5 if dV == 0 else 0.4)
        for row in tab[(tab[:, 0] <= 14) & (np.round(tab[:, 0] * 10) % 5 == 0)]:
            U = row[0]; res = solve(U, dV, x); x = tuple(res.x); d_, e_ = res.x; Z, P, n, eps = quantities(d_ * d_ + e_ * e_, d_ * d_ - e_ * e_, U, dV)
            worst["E"] = max(worst["E"], abs(res.fun - row[1])); worst["Z"] = max(worst["Z"], abs(Z - 0.5 * (row[2] + row[3]))); worst["P"] = max(worst["P"], abs(P - row[5])); worst["eps"] = max(worst["eps"], abs(eps - row[7]))
        print(f"dV={dV:4.1f} checked", flush=True)
    print("max |analytic reduction - numerical NotImpDimer(Ng=0)|:", {k: f"{v:.1e}" for k, v in worst.items()})
    uc = np.loadtxt("scan_ng0/Uc_ng0_vs_dV.txt"); f = 4 * t + 2 * np.sqrt(4 * t * t + uc[:, 0] ** 2)
    print("U_c closed form 4t+2sqrt(4t^2+dV^2) vs numerical extrapolation: max |diff| = %.1e over %d values of dV (0..5 and 10); at dV=2: %.4f vs %.4f, dV=10: %.4f vs %.4f" % (abs(f - uc[:, 1]).max(), len(uc), f[10], uc[10, 1], f[-1], uc[-1, 1]))
