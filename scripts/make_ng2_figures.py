"""Ng=2-only versions of the figures of doc/results_Ng2.tex, from saved data (no solver runs). Run from the repo root.
Writes merged_Ng/E_vs_U_dV{0,2}_ng2.png (as plot_merged_Ng.py without Ng=4) and merged_Ng/ionic_weight_vs_U_ng2.png
(as plot_ionic_weight.plot without Ng=4, from merged_Ng/ionic_weight_data.npz)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); sys.path.insert(0, "scan_ed_deltaV")
import plot_merged_Ng as pm

for dV in (0, 2):
    if dV == 0:
        E2 = pm.load_lowest("scan_ng2_not_imp/lowest_energy.txt"); ed = lambda U: pm.exact_dimer.solve(pm.t, U)[0]
    else:
        E2 = pm.load_lowest("scan_ng2_dV2_confirmed/lowest_energy.txt"); ed = lambda U: pm.ed_dV.E0_explicit(pm.t, U, 2.0 * dV)
    Us = np.array(sorted(E2)); e2 = np.array([E2[U] for U in Us]); eed = np.array([ed(U) for U in Us])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    ax[0].plot(Us, e2, "-", lw=1.8, color="tab:blue", label="gGA, $N_g=2$"); ax[0].plot(Us, eed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("$E_{var}/t$"); ax[0].legend(fontsize=8)
    ax[1].semilogy(Us, np.maximum(e2 - eed, 1e-12), "-", lw=1.8, color="tab:blue")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("$E_{var}-E_{ED}$ (error vs exact)")
    plt.tight_layout(); path = f"merged_Ng/E_vs_U_dV{dV}_ng2.png"; plt.savefig(path, dpi=150); plt.close(); print("wrote", path)

d = np.load("merged_Ng/ionic_weight_data.npz"); U = d["U"]; UC = {0: 6.56, 2: 8.34}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, dV in zip(axs, (0, 2)):
    ax.plot(U, d[f"ed_dV{dV}"], color="0.1", lw=2.2, label="ED (exact)")
    ax.plot(U, d[f"gga_dV{dV}_ng2"], color="#0072B2", lw=2.0, label=r"gGA, $N_g=2$")
    ax.axvline(UC[dV], color="0.55", lw=1.0, ls=":")
    ax.text(UC[dV] + 0.15, 0.93 if dV == 2 else 0.5, rf"$U_c/t={UC[dV]:.2f}$", color="0.35", fontsize=11, va="top")
    ax.set_xlabel(r"$U/t$"); ax.set_title(rf"$\Delta V={dV}$", fontsize=13); ax.set_xlim(0, 10); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right" if dV == 0 else "center right")
axs[0].set_ylabel(r"ionic weight $P_{\rm ion}=\langle n_{0\uparrow}n_{0\downarrow}\rangle+\langle n_{1\uparrow}n_{1\downarrow}\rangle$")
plt.tight_layout(); plt.savefig("merged_Ng/ionic_weight_vs_U_ng2.png", dpi=150); plt.close(fig); print("wrote merged_Ng/ionic_weight_vs_U_ng2.png")
