"""Ng=2: critical U_c/t of the gap closing vs the site-splitting dV (0..5, step 0.2), from scan_Uc_vs_dV/Uc_vs_dV.txt (scripts/Uc_from_scan.py), plus the single
point dV=10 from scan_Uc_dV10/Uc_dV10.txt (scripts/Uc_dV10.py), joined to the dV<=5 curve by a dashed straight segment (no points in between).
Writes merged_Ng/Uc_vs_dV.png. Run from the repo root."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.loadtxt("scan_Uc_vs_dV/Uc_vs_dV.txt"); dV, uc = d[:, 0], d[:, 1]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(dV, uc, "-o", color="#0072B2", lw=1.8, ms=4)
d10 = np.loadtxt("scan_Uc_dV10/Uc_dV10.txt"); ax.plot([dV[-1], d10[0]], [uc[-1], d10[1]], "--", color="#0072B2", lw=1.4)   # straight guide between dV=5 and 10 (no data in between)
ax.plot([d10[0]], [d10[1]], "o", color="#0072B2", mfc="white", ms=7, mew=1.8)
ax.annotate(rf"$U_c/t={d10[1]:.2f}$", (d10[0], d10[1]), textcoords="offset points", xytext=(0, -26), fontsize=10, color="0.3", ha="center")
for x in (0.0, 2.0):
    i = int(np.argmin(abs(dV - x))); ax.annotate(rf"$U_c/t={uc[i]:.2f}$", (dV[i], uc[i]), textcoords="offset points", xytext=(8, -14), fontsize=10, color="0.3")
ax.set_xlabel(r"$\Delta V/t$"); ax.set_ylabel(r"$U_c/t$ (spectral gap closes)"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 25)
plt.tight_layout(); plt.savefig("merged_Ng/Uc_vs_dV.png", dpi=150); print("wrote merged_Ng/Uc_vs_dV.png")
