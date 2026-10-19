"""3D "ridgeline" versions of the eqp_vs_U.png plots: the spectral function A(omega) itself,
as a stack of solid, Lorentzian-broadened curves over U/t = 0.1 .. 10.0 (depth axis), instead
of the raw H_qp eigenvalues -- this avoids the confusing back-and-forth of individual pole
lines through level crossings (see the previous version of this script).

For every U-slice, A(omega) of the gGA solution is exactly the routine of
gga_dimer.compute_spectral_function (same (R, lambda), eta broadening) used throughout this
project; it is drawn as a solid filled curve. The EXACT (ED) A(omega) of the half-filled
Hubbard dimer, from the closed-form Lehmann-representation result of
/home/marm314/C_MEJUTO/gGut_notes/gGut/hubbard_dimer/gGut_notes/gGut.tex
(Eq. eq:Aw_exact: poles at +-(Omega/2-t) weight 1+4t/Omega, +-(Omega/2+t) weight 1-4t/Omega,
Omega = sqrt(U^2+16t^2)), with the SAME eta, is overlaid as a thin red line on top of each slice.

Output: eqp_vs_U_3d.png next to each eqp_vs_U.png (not modified).
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

import gga_dimer as g

t = 1.0
ETA = 0.1   # same broadening as gGut.tex Fig. "Spectral function" (eta = 0.1 t)
OMEGA_MAX = 8.0
DOMEGA = 0.05


def ed_Aomega(omega, U, t=1.0, eta=ETA):
    Omega = np.sqrt(U ** 2 + 16.0 * t ** 2)
    inner, outer = Omega / 2.0 - t, Omega / 2.0 + t
    w_in, w_out = 1.0 + 4.0 * t / Omega, 1.0 - 4.0 * t / Omega
    lor = lambda x: eta / (np.pi * (x ** 2 + eta ** 2))
    A = np.zeros_like(omega)
    for nu in (+1.0, -1.0):
        A += w_in * lor(omega - nu * inner) + w_out * lor(omega - nu * outer)
    return A


def load_R_lambda(d):
    R = np.loadtxt(glob.glob(os.path.join(d, "*_R.dat"))[0])
    lam = np.loadtxt(glob.glob(os.path.join(d, "*_lambda.dat"))[0])
    Neff = R.shape[1]
    return [R[0], R[1]], [lam[:Neff], lam[Neff:]]


def branch_of(path_a, path_b):
    def energies(path):
        out = {}
        for f in glob.glob(os.path.join(path, "U_*", "output.out")):
            line = open(f).readlines()[-1]
            m_u = re.search(r"U/t=([\d.]+)", line)
            m_e = re.search(r"E_var/t=(-?[\d.]+)", line)
            out[round(float(m_u.group(1)), 1)] = float(m_e.group(1))
        return out
    Ea, Eb = energies(path_a), energies(path_b)
    return {u: (path_a if Ea[u] <= Eb[u] else path_b) for u in sorted(Ea)}


def ridge_plot(get_dir_for_U, Us, title, png_path, ed_fn=ed_Aomega):
    """ed_fn(omega, U, t, eta): the exact-ED A(omega) overlay; defaults to the Delta V=0
    closed form above, but e.g. plot_eqp_3d_dV2.py passes a Delta V=2 one instead."""
    fig = plt.figure(figsize=(8.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.viridis
    zmax = 0.0
    for k, U in enumerate(Us):
        d = get_dir_for_U(U)
        R, lam = load_R_lambda(d)
        omega_grid, A = g.compute_spectral_function(R[0], R[1], lam[0], lam[1], t, OMEGA_MAX, ETA, DOMEGA)
        A_ed = ed_fn(omega_grid, U, t, ETA)
        zmax = max(zmax, A.max(), A_ed.max())

        color = cmap(k / max(1, len(Us) - 1))
        verts = [(omega_grid[0], 0.0)] + list(zip(omega_grid, A)) + [(omega_grid[-1], 0.0)]
        poly = PolyCollection([verts], facecolors=[color], edgecolors="k", linewidths=0.4, alpha=0.85)
        ax.add_collection3d(poly, zs=U, zdir="y")
        ax.plot(omega_grid, [U] * len(omega_grid), A_ed, color="red", lw=0.9, alpha=0.9)

    ax.set_xlim(-OMEGA_MAX, OMEGA_MAX)
    ax.set_ylim(min(Us), max(Us))
    ax.set_zlim(0, 1.08 * zmax)
    ax.set_xlabel("quasiparticle energy / t")
    ax.set_ylabel("U/t")
    ax.set_zlabel(rf"A($\omega$)  ($\eta$={ETA:g})")
    ax.set_title(title)

    # no background mesh/panes at all
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")

    ax.view_init(elev=28, azim=-50)
    handles = [Patch(facecolor=cmap(0.5), edgecolor="k", alpha=0.85, label="gGA A(omega)"),
               Line2D([0], [0], color="red", lw=1.2, label="ED A(omega) (exact)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {png_path}")


Us_ridge = [round(u, 1) for u in np.arange(0.1, 10.01, 1.0)]  # 0.1, 1.0, 2.0, ..., 10.0

if __name__ == "__main__":
    # 1) scan_ng2_imp: gga_dimer.py, lowest of the backward/forward branches
    br = branch_of("scan_ng2_imp/backward", "scan_ng2_imp/forward")
    ridge_plot(lambda U: br[U] + f"/U_{U:.1f}", Us_ridge,
               "Spectral function vs U (Ng=2, minimum-energy branch)",
               "scan_ng2_imp/eqp_vs_U_3d.png")

    # 2) scan_ng2_not_imp: direct minimization, lowest_energy.txt records the branch
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            p = line.split()
            br2[round(float(p[0]), 1)] = p[2]
    ridge_plot(lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}", Us_ridge,
               "Spectral function, lowest-energy solution (Ng=2)",
               "scan_ng2_not_imp/eqp_vs_U_3d.png")

    # 3) scan_ng2_not_imp_sol_at_imp: gga_dimer.py seeded by the direct-minimization solutions
    ridge_plot(lambda U: f"scan_ng2_not_imp_sol_at_imp/U_{U:.1f}", Us_ridge,
               "Spectral function of gga_dimer.py (Ng=2),\nseeded with the direct-minimization solutions",
               "scan_ng2_not_imp_sol_at_imp/eqp_vs_U_3d.png")

    # 4) scan_ng4_not_imp
    ridge_plot(lambda U: f"scan_ng4_not_imp/U_{U:.1f}", Us_ridge,
               "Spectral function vs U (Ng=4)",
               "scan_ng4_not_imp/eqp_vs_U_3d.png")
