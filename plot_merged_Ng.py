"""Merged Ng=2 / Ng=4 energy and error-vs-ED plots, one figure per deltaV (0 and 2), built only
from already-saved scan data (no solver re-runs):
  deltaV=0: scan_ng2_not_imp/lowest_energy.txt (Ng=2), scan_ng4_not_imp/summary.txt (Ng=4);
  deltaV=2: scan_ng{2,4}_dV2_confirmed/lowest_energy.txt (cross-checked mutual solutions).
Exact energies: exact_dimer.solve (deltaV=0) and scan_ed_deltaV/make_plot.E0_explicit with
deltaV_ed = 2*deltaV (deltaV=2). Writes merged_Ng/E_vs_U_dV0.png and merged_Ng/E_vs_U_dV2.png.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "scan_ed_deltaV"))
import exact_dimer
import make_plot as ed_dV

t = 1.0
OUT = os.path.join(here, "merged_Ng")


def load_lowest(path):
    d = {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            d[round(float(p[0]), 1)] = float(p[1])
    return d


def load_summary_ng4(path):
    d = {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            d[round(float(p[0]), 1)] = float(p[1])
    return d


def make(dV):
    if dV == 0:
        E2 = load_lowest(os.path.join(here, "scan_ng2_not_imp", "lowest_energy.txt"))
        E4 = load_summary_ng4(os.path.join(here, "scan_ng4_not_imp", "summary.txt"))
        ed = lambda U: exact_dimer.solve(t, U)[0]
    else:
        E2 = load_lowest(os.path.join(here, "scan_ng2_dV2_confirmed", "lowest_energy.txt"))
        E4 = load_lowest(os.path.join(here, "scan_ng4_dV2_confirmed", "lowest_energy.txt"))
        ed = lambda U: ed_dV.E0_explicit(t, U, 2.0 * dV)
    Us = np.array(sorted(set(E2) & set(E4)))
    e2 = np.array([E2[U] for U in Us]); e4 = np.array([E4[U] for U in Us])
    eed = np.array([ed(U) for U in Us])

    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    ax[0].plot(Us, e2, "-", lw=1.8, color="tab:blue", label="gGA, $N_g=2$")
    ax[0].plot(Us, e4, ":", lw=2.6, color="red", label="gGA, $N_g=4$")
    ax[0].plot(Us, eed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("$E_{var}/t$")
    ax[0].legend(fontsize=8)
    r2, r4 = e2 - eed, e4 - eed
    ax[1].semilogy(Us, np.maximum(r2, 1e-12), "-", lw=1.8, color="tab:blue", label="$N_g=2$")
    ax[1].semilogy(Us, np.maximum(r4, 1e-12), ":", lw=2.6, color="red", label="$N_g=4$")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("$E_{var}-E_{ED}$ (error vs exact)")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT, f"E_vs_U_dV{dV}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"dV={dV}: wrote {path}; max err Ng2={r2.max():.3e} Ng4={r4.max():.3e}; "
          f"min err Ng2={r2.min():.2e} Ng4={r4.min():.2e}; max|E4-E2|={np.abs(e4-e2).max():.2e}; "
          f"min(E-E_ED)<0? {min(r2.min(), r4.min()) < 0}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for dV in (0, 2):
        make(dV)
