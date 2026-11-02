"""Ng=0 version of the 3D A(omega) figure of doc/results.tex (same style as plot_eqp_3d_pos.py: gGA A(omega) filled, ED red, omega>=0), from the ANALYTIC Ng=0 solution
(scripts/ng0_analytic.py: closed form at dV=0, two-variable reduction at dV!=0). Writes merged_Ng/ng0_3d_dV0.png and ng0_3d_dV2.png. Run from the repo root."""
import sys, numpy as np
sys.path.insert(0, "scripts")
import plot_eqp_3d as p
import plot_eqp_3d_dV2 as q
from ng0_analytic import ng0_state
t = 1.0
def load_ng0(key):                     # key = "dV|U": R and lambda of the analytic Ng=0 solution (scripts/ng0_analytic.py): R0=R1=sqrt(Z), lambda_0=-lambda_1
    dV, U = (float(x) for x in key.split("|")); a = ng0_state(U, dV)
    return [np.array([np.sqrt(a["Z"])]), np.array([np.sqrt(a["Z"])])], [np.array([[a["lam"]]]), np.array([[-a["lam"]]])]
p.load_R_lambda = load_ng0
for dV, ed in ((0.0, p.ed_Aomega), (2.0, q.ed_Aomega_dV2)):
    p.ridge_plot(lambda U, dV=dV: f"{dV}|{U}", p.Us_ridge, f"Ng=0, dV={dV:g}", f"merged_Ng/ng0_3d_dV{int(dV)}.png", ed_fn=ed, omega_min=0.0)
