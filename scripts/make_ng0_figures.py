"""Figures with the Ng=0 (ordinary Gutzwiller) case added to the Ng=2 ones, from saved data (Ng=2) and the ANALYTIC Ng=0 solution (scripts/ng0_analytic.py: closed form of\ngGut_notes.pdf Sec. III D at dV=0, two-variable reduction at dV!=0); no Ng=0 solver data is used. Run from the repo root. Writes merged_Ng/E_vs_U_dV{0,2}_ng02.png, poles_ng0.png, ionic_weight_vs_U_ng02.png, Uc_vs_dV_ng02.png."""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); sys.path.insert(0, "scan_ed_deltaV")
import plot_merged_Ng as pm
t = 1.0; C0, C2 = "#D55E00", "#0072B2"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
from ng0_analytic import ng0_state
def ng0(dV):                           # columns as scan_ng0/table: U E Z0 Z1 docc Pion eps0 eps1 w0 w1, from the analytic solution
    rows = []
    for U in np.round(np.arange(0.1, 15.01, 0.1), 1):
        a = ng0_state(U, dV); rows.append([U, a["E"], a["Z"], a["Z"], a["docc"], a["c"], -a["eps"], a["eps"], a["w"], a["w"]])
    return np.array(rows)
UC0 = np.array([[x, 4 * t + 2 * np.sqrt(4 * t * t + x * x)] for x in [round(0.2 * k, 1) for k in range(26)] + [10.0]])      # closed form
UC2 = np.loadtxt("scan_Uc_vs_dV/Uc_vs_dV.txt")
uc0 = lambda dV: UC0[np.argmin(abs(UC0[:, 0] - dV)), 1]
# energies
for dV in (0, 2):
    E2 = pm.load_lowest("scan_ng2_not_imp/lowest_energy.txt" if dV == 0 else "scan_ng2_dV2_confirmed/lowest_energy.txt")
    ed = (lambda U: pm.exact_dimer.solve(t, U)[0]) if dV == 0 else (lambda U: pm.ed_dV.E0_explicit(t, U, 2.0 * dV))
    r = ng0(dV); Us = np.array([u for u in sorted(E2) if u <= 10.0]); e2 = np.array([E2[u] for u in Us]); eed = np.array([ed(u) for u in Us]); e0 = np.interp(Us, r[:, 0], r[:, 1])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    ax[0].plot(Us, e0, "-", lw=1.8, color=C0, label="gGA, $N_g=0$"); ax[0].plot(Us, e2, "-", lw=1.8, color=C2, label="gGA, $N_g=2$"); ax[0].plot(Us, eed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("$E_{var}/t$"); ax[0].legend(fontsize=8)
    ax[1].semilogy(Us, np.maximum(e0 - eed, 1e-12), color=C0, lw=1.8); ax[1].semilogy(Us, np.maximum(e2 - eed, 1e-12), color=C2, lw=1.8)
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("$E_{var}-E_{ED}$ (error vs exact)"); plt.tight_layout(); plt.savefig(f"merged_Ng/E_vs_U_dV{dV}_ng02.png", dpi=150); plt.close()
# poles
import make_plot as ed_dV
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
for ax, dV in zip(axs, (0, 2)):
    r = ng0(dV); m = r[:, 0] <= 12; ax.plot(r[m, 0], r[m, 7], "-", color=C0, lw=2.2, label="gGA, $N_g=0$", zorder=2); ax.scatter(r[m, 0][::5], r[m, 7][::5], s=10 + 22 * r[m, 9][::5], color=C0, edgecolors="white", lw=0.5, zorder=4)
    Ue = np.arange(0.1, 12.01, 0.1); P = [(u, o, w) for u in Ue for o, w in zip(*ed_dV.analytic_poles(t, u, 2.0 * dV)) if o > 0]
    ax.scatter([p[0] for p in P], [p[1] for p in P], s=[2 + 14 * p[2] for p in P], color="black", label="ED", zorder=3)
    ax.axvline(uc0(dV), color="0.55", ls=":", lw=1.0); ax.text(uc0(dV) + 0.15, ax.get_ylim()[1] * 0.2, rf"$U_c/t={uc0(dV):.2f}$", color="0.35", fontsize=10, rotation=90, va="center")
    ax.set_xlabel(r"$U/t$"); ax.set_title(rf"$\Delta V={dV}$"); ax.set_xlim(0, 12); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99 if dV == 0 else 0.64))
axs[0].set_ylabel(r"positive pole energy $\omega/t$ (area $\propto$ weight)"); plt.tight_layout(); plt.savefig("merged_Ng/poles_ng0.png", dpi=150); plt.close()
# P_ion
d = np.load("merged_Ng/ionic_weight_data.npz"); U = d["U"]; fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, dV in zip(axs, (0, 2)):
    r = ng0(dV); ax.plot(U, d[f"ed_dV{dV}"], color="0.1", lw=2.2, label="ED (exact)"); ax.plot(U, np.interp(U, r[:, 0], r[:, 5]), color=C0, lw=2.0, label=r"gGA, $N_g=0$")
    ax.plot(U, d[f"gga_dV{dV}_ng2"], color=C2, lw=2.0, label=r"gGA, $N_g=2$")
    for u, c in ((uc0(dV), C0), (UC2[np.argmin(abs(UC2[:, 0] - dV)), 1], C2)): ax.axvline(u, color=c, lw=1.0, ls=":")
    ax.set_xlabel(r"$U/t$"); ax.set_title(rf"$\Delta V={dV}$", fontsize=13); ax.set_xlim(0, 10); ax.set_ylim(0, 1.0); ax.legend(frameon=False, loc="upper right" if dV == 0 else "center right")
axs[0].set_ylabel(r"ionic weight $P_{\rm ion}=\langle n_{0\uparrow}n_{0\downarrow}\rangle+\langle n_{1\uparrow}n_{1\downarrow}\rangle$"); plt.tight_layout(); plt.savefig("merged_Ng/ionic_weight_vs_U_ng02.png", dpi=150); plt.close()
# U_c vs dV
d10 = np.loadtxt("scan_Uc_dV10/Uc_dV10.txt"); fig, ax = plt.subplots(figsize=(7.0, 4.4)); m = UC0[:, 0] <= 5
x = np.linspace(0, 10, 200); ax.plot(x, 4 * t + 2 * np.sqrt(4 * t * t + x * x), "-", color=C0, lw=1.8, label=r"$N_g=0$: $4t+2\sqrt{4t^2+\Delta V^2}$"); ax.plot(UC0[:, 0], UC0[:, 1], "o", color=C0, ms=3.5)
ax.annotate(r"$24.40$", (10, 4 * t + 2 * np.sqrt(4 * t * t + 100)), textcoords="offset points", xytext=(0, 9), fontsize=10, color="0.3", ha="center")
for u, c, lab, p10 in ((UC2, C2, r"$N_g=2$", d10),):
    ax.plot(u[:, 0], u[:, 1], "-o", color=c, lw=1.8, ms=4, label=lab); ax.plot([u[-1, 0], p10[0]], [u[-1, 1], p10[1]], "--", color=c, lw=1.4)
    ax.plot([p10[0]], [p10[1]], "o", color=c, mfc="white", ms=7, mew=1.8); ax.annotate(rf"${p10[1]:.2f}$", (p10[0], p10[1]), textcoords="offset points", xytext=(0, -18), fontsize=10, color="0.3", ha="center")
ax.annotate(r"$8$", (0, 8), textcoords="offset points", xytext=(6, 4), fontsize=10, color="0.3"); ax.annotate(r"$6.56$", (0, 6.56), textcoords="offset points", xytext=(6, -14), fontsize=10, color="0.3")
ax.set_xlabel(r"$\Delta V/t$"); ax.set_ylabel(r"$U_c/t$"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 27); ax.legend(frameon=False, loc="upper left"); plt.tight_layout(); plt.savefig("merged_Ng/Uc_vs_dV_ng02.png", dpi=150); plt.close()
print("wrote figures")
