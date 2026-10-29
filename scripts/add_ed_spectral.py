"""Overlay the exact (ED) spectral poles of the half-filled Hubbard dimer on the
eqp_vs_U.png plots, from the closed-form Lehmann-representation result of
/home/marm314/C_MEJUTO/gGut_notes/gGut/hubbard_dimer/gGut_notes/gGut.tex
(Appendix "Exact spectral function of the dimer", Eq. eq:Aw_exact):

    Omega = sqrt(U^2 + 16 t^2)
    poles at  +-(Omega/2 - t)   weight 1 + 4t/Omega   ("quasi-particle" / inner poles)
              +-(Omega/2 + t)   weight 1 - 4t/Omega   ("satellite" / outer poles)

These are exact many-body addition/removal energies (poles of the Lehmann Green's
function), not eigenvalues of a one-body H_qp, so they are overlaid for comparison
rather than merged with the gGA eqp branches. Does not modify the underlying
eqp_vs_U.txt data; only regenerates the eqp_vs_U.png next to it, same title as before.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = 1.0

TARGETS = [
    ("scan_ng2_imp/eqp_vs_U.txt", "scan_ng2_imp/eqp_vs_U.png",
     "Quasiparticle spectrum vs U (Ng=2, minimum-energy branch)"),
    ("scan_ng2_not_imp/eqp_vs_U.txt", "scan_ng2_not_imp/eqp_vs_U.png",
     "Quasiparticle energies, lowest-energy solution (Ng=2)"),
    ("scan_ng2_not_imp_sol_at_imp/eqp_vs_U.txt", "scan_ng2_not_imp_sol_at_imp/eqp_vs_U.png",
     "Quasiparticle energies of gga_dimer.py (Ng=2),\nseeded with the direct-minimization solutions"),
    ("scan_ng4_not_imp/eqp_vs_U.txt", "scan_ng4_not_imp/eqp_vs_U.png",
     "Quasiparticle energies vs U (Ng=4)"),
]


def ed_poles(U, t=1.0):
    Omega = np.sqrt(U ** 2 + 16.0 * t ** 2)
    return Omega / 2.0 - t, Omega / 2.0 + t   # inner (quasi-particle), outer (satellite)


for txt_path, png_path, title in TARGETS:
    data = np.loadtxt(txt_path)
    data = data[np.argsort(data[:, 0])]
    U, eq = data[:, 0], data[:, 1:]

    # the half-filled dimer is particle-hole symmetric, so the (ascending) eigenvalues of
    # H_qp come in +-E pairs, column i with column n-1-i; give each pair one color (checked
    # against eqp_vs_U.txt: holds to ~1e-3 for the outer, physical levels in every file here;
    # the near-zero inner levels of Ng=4 are a reconstruction artifact -- see add_ed_spectral.py
    # docstring reference -- and do not pair cleanly by fixed column index, so their color
    # pairing there is only approximate)
    n = eq.shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(6.4, 4.8))
    for i in range(n):
        pair = min(i, n - 1 - i)
        plt.plot(U, eq[:, i], "-", lw=1.1, color=colors[pair % len(colors)])
    inner, outer = ed_poles(U, t)
    plt.plot(U, inner, "k--", lw=1.5, label="ED quasi-particle poles")
    plt.plot(U, -inner, "k--", lw=1.5)
    plt.plot(U, outer, "k:", lw=1.5, label="ED satellite poles")
    plt.plot(U, -outer, "k:", lw=1.5)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t")
    plt.ylabel("quasiparticle energy / t")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"updated {png_path}")
