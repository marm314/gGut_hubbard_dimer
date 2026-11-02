"""Analytic Ng=0 solution of the half-filled dimer (no numerical scan): dV=0 closed form (gGut_notes.pdf Sec. III D), dV!=0 the two-variable reduction of
scripts/ng0_dV_analytic.py (minimum of E(delta,eps), or the trivial solution Z=0, E=-U above U_c = 4t+2 sqrt(4t^2+dV^2)).
ng0_state(U,dV) -> dict(E, Z, c (=P_ion), m (n_favored=1+m), lam (lambda_0=-lambda_1), eps (qp level >0), w (weight of each level = 2Z), docc=c/2)."""
import numpy as np
from scipy.optimize import minimize
import ng0_dV_analytic as A
t = 1.0
def ng0_state(U, dV):
    if dV == 0:
        ok = U < 8; Z = max(1 - (U / 8) ** 2, 0.0); c = (1 / 2 - U / 16) if ok else 0.0; m = 0.0; E = (-2 * t - U / 2 - U ** 2 / 32) if ok else -U
    else:
        if U >= 4 * t + 2 * np.sqrt(4 * t * t + dV * dV): c = m = Z = 0.0; E = -U
        else:
            r = A.solve(U, dV, (0.5, 0.4)); d, e = r.x; E = r.fun; c, m = d * d + e * e, d * d - e * e
            rr = np.sqrt(max(c * c - m * m, 0)); Z = 2 * (1 - c) * (c + rr) / (1 - m * m)
            if E > -U: c = m = Z = 0.0; E = -U          # trivial solution if it is lower
    lam = t * Z * m / np.sqrt(1 - m * m) if Z > 0 else 0.0; eps = t * Z / np.sqrt(1 - m * m)
    return dict(E=E, Z=Z, c=c, m=m, lam=lam, eps=eps, w=2 * Z, docc=c / 2)
