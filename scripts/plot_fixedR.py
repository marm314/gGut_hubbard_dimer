"""Self-consistent fixed-|R| study (psi_fixedR.py, data ../psi_scan/Rfix_ng<Ng>_dV0_U<U>_norm.txt): P_ion of Psi_G and the gGA energy change vs s (|R_i| = s |R_i^sol|)
for Ng=0 and Ng=2 (dV=0). Output ../merged_Ng/Rfix_pion_vs_R.png. Points with overlap < 0.95 with the exact ground state are left out of the P_ion panels."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = {2: "#1b9e77", 4: "#d95f02", 5: "#7570b3", 6: "#e7298a", 7: "#66a61e", 8: "#666666"}
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex="col")
for k, (ng, Us) in enumerate(((0, (2, 4, 5, 6, 7)), (2, (2, 4, 5, 6, 7, 8)))):
    for U in Us:
        a = np.loadtxt(f"../psi_scan/Rfix_ng{ng}_dV0_U{U}.0_norm.txt", ndmin=2); a = a[np.argsort(-a[:, 0])]
        e1 = a[abs(a[:, 0] - 1) < 1e-9][0, 1]; ok = a[:, 6] >= 0.95
        axs[0, k].plot(a[ok, 0], a[ok, 4], "o-", ms=3.5, color=cols[U], label=f"$U/t$={U}")
        axs[1, k].plot(a[:, 0], a[:, 1] - e1, "o-", ms=3.5, color=cols[U], label=f"$U/t$={U}")
    axs[0, k].set_title(f"Ng={ng}"); axs[0, k].set_ylabel("$P_{ion}(\\Psi_G)$"); axs[1, k].set_ylabel("$E_{var}(s)-E_{var}(1)$ (gGA functional)")
    axs[1, k].set_xlabel("$s$,  $|R_i|=s\\,|R_i^{\\rm sol}|$"); axs[0, k].legend(fontsize=8); axs[0, k].invert_xaxis()
axs[0, 0].set_yscale("log"); axs[0, 0].set_ylim(1e-3, 0.6)
plt.tight_layout(); plt.savefig("../merged_Ng/Rfix_pion_vs_R.png", dpi=130)
