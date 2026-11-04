"""Role of R in the ionic/covalent character of Psi_G (data: psi_R_modify.py, psi_scan/Rmod_*.txt). Output ../merged_Ng/Rmod_role_of_R.png
(a) Ng=0, dV=0, U/t=4: P_ion and E_Psi vs R/R_sol.  (b) Ng=2, dV=0, U/t=4: R_i -> s R_i, only in H_qp (A) or everywhere with the embedding re-solved (B).
(c) Ng=2, norm |R_i| kept: P_ion vs the weight of R on the HOMO/LUMO-like orbitals (Ng=0 dashed: x = R^2)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gutzwiller_psi as G
from plot_Rmod import share

def load(f, mode):
    a = np.array([[float(x) for x in l.split()[1:]] for l in open(f) if l[0] == mode])   # s norm Pion cov ion_s ion_a E ovl ...
    return a[np.argsort(a[:, 0])]

PED = 0.1464; EED = -4.82843
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axs[0]; a = load("../psi_scan/Rmod_ng0_dV0_U4.0_both.txt", "B")
ax.plot(a[:, 0], a[:, 2], "o-", color="#d95f02", label="$P_{ion}(\\Psi_G)$"); ax.axhline(PED, color="k", ls=":", label="$P_{ion}$ exact")
ax.axvline(1, color="gray", lw=0.8); ax.set_xlabel("$R/R_{\\rm gGA}$"); ax.set_ylabel("$P_{ion}$"); ax.set_title("(a) Ng=0, $\\Delta V$=0, $U/t$=4")
ax2 = ax.twinx(); ax2.plot(a[:, 0], a[:, 6] - EED, "s--", color="#1b9e77", ms=4, label="$E_\\Psi-E_{FCI}$"); ax2.set_ylabel("$E_\\Psi-E_{FCI}$", color="#1b9e77"); ax2.set_ylim(-0.02, 0.9)
ax.annotate("covalent", (0.05, 0.01), color="gray"); ax.annotate("ionic", (1.3, 0.18), color="gray")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
ax = axs[1]
for mode, c, lab in (("A", "#7570b3", "A: $R$ only in $H_{qp}$"), ("B", "#d95f02", "B: $R$ everywhere, embedding re-solved")):
    a = load("../psi_scan/Rmod_ng2_dV0_U4.0_both.txt", mode); ax.plot(a[:, 0], a[:, 2], "o-", color=c, label=lab)
ax.axhline(PED, color="k", ls=":", label="exact"); ax.axvline(1, color="gray", lw=0.8)
ax.set_xlabel("$s$,  $R_i\\to sR_i$ (|R| leaves 1)"); ax.set_ylabel("$P_{ion}$"); ax.set_title("(b) Ng=2, $\\Delta V$=0, $U/t$=4"); ax.legend(fontsize=8)
ax = axs[2]; cols = {2.0: "#1b9e77", 4.0: "#d95f02", 6.0: "#7570b3"}
for dv, ls in (("dV0", "--"), ("dV2", "-")):
    for U, c in cols.items():
        sh, _ = share(dv, U); a = load(f"../psi_scan/Rmod_ng2_{dv}_U{U:.1f}_both_rot.txt", "A")
        m = a[:, 0] <= 1.0 + 1e-9 if dv == "dV0" else np.ones(len(a), bool)           # dV=0: below the solution only (above it: level crossing, not shown)
        ax.plot((a[:, 0] ** 2 * sh)[m], a[:, 2][m], "o" + ls, color=c, ms=4, label=f"Ng=2 $\\Delta V$={dv[2:]}, $U/t$={U:g}")
        ax.plot([sh], [a[np.argmin(abs(a[:, 0] - 1)), 2]], "k*", ms=9)
a = load("../psi_scan/Rmod_ng0_dV0_U4.0_both.txt", "B"); ax.plot(0.75 * a[:, 0] ** 2, a[:, 2], "s:", color="gray", ms=4, label="Ng=0 $U/t$=4 ($x=R^2$)")
ax.set_xlabel("weight of $R$ on HOMO/LUMO, $|R_{HL}|^2$ ($|R_i|$ kept)"); ax.set_ylabel("$P_{ion}$"); ax.set_title("(c) Ng=2, $R$ moved to the ghosts"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("../merged_Ng/Rmod_role_of_R.png", dpi=130)
