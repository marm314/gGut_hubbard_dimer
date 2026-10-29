"""Delta V/t=2.0 analogue of plot_eqp_3d.py's ridgeline plots, for the three new scans
scan_ng2_imp_dV2/, scan_ng2_not_imp_dV2/ and scan_ng4_not_imp_dV2/ (see scan_ng2_imp_dV2_U.py,
scan_ng2_not_imp_dV2_U.py, scan_ng4_not_imp_dV2_U.py). gGA's own A(omega) (gga_dimer.compute_spectral_function) needs
no change -- it never sees deltaV directly, only through the already-converged (R, lambda)
it is handed (see gga_dimer.qp_state's docstring: deltaV enters only through H_loc's own
chemical potential, not H_qp). Only the EXACT (ED) A(omega) overlay changes: it now uses the
Delta V=2 closed form of scan_ed_deltaV/ED_deltaV.tex (scan_ed_deltaV/make_plot.py's
analytic_poles, itself cross-checked there to machine precision against
exact_dimer_deltaV.py's numerical ED), not the Delta V=0 closed form of plot_eqp_3d.ed_Aomega.

Output: eqp_vs_U_3d.png next to each eqp_vs_U.png (not modified).

IMPORTANT: exact_dimer_deltaV.py's deltaV and gga_dimer.py's deltaV are NOT the same
physical parameter -- exact_dimer_deltaV.py's one-body term is -mu(n0+n1)+(deltaV/2)(n0-n1)
with mu=U/2 (site-0 coefficient -U/2+deltaV/2), while gga_dimer.py's (C++-validated)
convention is mu_0=U/2+deltaV, mu_1=U/2-deltaV (site-0 coefficient -U/2-deltaV). Matching
coefficients gives deltaV_ed = -2*deltaV_gga; since the closed form is an even function of
deltaV (the model's site-exchange symmetry), deltaV_ed = +2*deltaV_gga works just as well
and is used below (DELTA_V_ED). Getting this wrong (i.e. passing DELTA_V_GGA directly to
the ED closed form) was caught via scan_ng2_not_imp_dV2_U.py's E_vs_U.png: the gGA
variational energy came out BELOW the (wrongly-converted) ED ground-state energy for
U/t up to ~5.5, by as much as 1.6 -- impossible for a genuine variational upper bound.
"""
import os
import sys

import numpy as np

from plot_eqp_3d import load_R_lambda, ridge_plot, t, ETA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scan_ed_deltaV"))
import make_plot as ed_dV  # scan_ed_deltaV/make_plot.py

DELTA_V_GGA = 2.0
DELTA_V_ED = 2.0 * DELTA_V_GGA


def ed_Aomega_dV2(omega, U, t=1.0, eta=ETA):
    poles, weights = ed_dV.analytic_poles(t, U, DELTA_V_ED)
    lor = lambda x: eta / (np.pi * (x ** 2 + eta ** 2))
    A = np.zeros_like(omega)
    for om_i, w_i in zip(poles, weights):
        A += w_i * lor(omega - om_i)
    return A


Us_ridge = [round(u, 1) for u in np.arange(0.1, 10.01, 1.0)]  # 0.1, 1.0, 2.0, ..., 10.0


def load_branches(lowest_energy_path):
    br = {}
    for line in open(lowest_energy_path):
        if not line.startswith("#") and line.strip():
            p = line.split()
            br[round(float(p[0]), 1)] = p[2]
    return br


if __name__ == "__main__":
    # 1) scan_ng2_imp_dV2: gga_dimer.py, lowest of its forward/backward branches
    br1 = load_branches("scan_ng2_imp_dV2/lowest_energy.txt")
    ridge_plot(lambda U: f"scan_ng2_imp_dV2/{br1[U]}/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$)",
               "scan_ng2_imp_dV2/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)

    # 2) scan_ng2_not_imp_dV2: direct minimization, lowest of its forward/backward branches
    br2 = load_branches("scan_ng2_not_imp_dV2/lowest_energy.txt")
    ridge_plot(lambda U: f"scan_ng2_not_imp_dV2/{br2[U]}/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$)",
               "scan_ng2_not_imp_dV2/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)

    # 3) scan_ng4_not_imp_dV2
    ridge_plot(lambda U: f"scan_ng4_not_imp_dV2/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$)",
               "scan_ng4_not_imp_dV2/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)

    # 4) scan_ng4_imp_dV2: gga_dimer.py, lowest of its forward/backward branches
    br4 = load_branches("scan_ng4_imp_dV2/lowest_energy.txt")
    ridge_plot(lambda U: f"scan_ng4_imp_dV2/{br4[U]}/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$)",
               "scan_ng4_imp_dV2/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)

    # 5) scan_ng4_dV2_confirmed: cross_check_ng4_dV2.py's mutually-confirmed winner per U
    # (single directory per U, no branch lookup needed)
    ridge_plot(lambda U: f"scan_ng4_dV2_confirmed/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$, cross-checked mutual solution)",
               "scan_ng4_dV2_confirmed/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)

    # 6) scan_ng2_dV2_confirmed: cross_check_ng2_dV2.py's mutually-confirmed winner per U
    ridge_plot(lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", Us_ridge,
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$, cross-checked mutual solution)",
               "scan_ng2_dV2_confirmed/eqp_vs_U_3d.png", ed_fn=ed_Aomega_dV2)
