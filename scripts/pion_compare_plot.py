"""P_ion from the impurity double occupancies (sum over fragments) vs P_ion of the normalized Psi_G; parses psi_scan/pion_compare_*.log."""
import re, glob, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = {}
for f in sorted(glob.glob("../psi_scan/pion_compare_*.log")):
    key = None
    for l in open(f):
        m = re.match(r"=== (dV\d) Ng=(\d) U/t=([\d.]+)", l)
        if m: key = (m[1], int(m[2]), float(m[3])); nrm = None
        m = re.search(r"<Psi\|Psi>=([\d.]+)", l)
        if m: nrm = float(m[1])
        m = re.search(r"P_ion\(Psi\)=([\d.]+)\s+P_ion\(ED\)=([\d.]+)\s+P_ion\(fragments\)=([\d.]+)", l)
        if m and key: rows[key] = (float(m[3]), float(m[1]), float(m[2]), nrm)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for (dv, ng), c, mk in [(("dV0", 2), "C0", "o"), (("dV0", 4), "C1", "s"), (("dV2", 2), "C2", "^"), (("dV2", 4), "C3", "v")]:
    pts = sorted((k[2], v) for k, v in rows.items() if k[:2] == (dv, ng) and 1e-6 < v[3] < 50)
    if not pts: continue
    U = [p[0] for p in pts]; fr = [p[1][0] for p in pts]; ps = [p[1][1] for p in pts]; ex = [p[1][2] for p in pts]
    ax[0].plot(fr, ps, mk, color=c, label=f"{dv} Ng={ng}")
    ax[1].plot(U, [a - b for a, b in zip(fr, ex)], mk + "-", color=c, mfc="none", label=f"fragments - ED, {dv} Ng={ng}")
    ax[1].plot(U, [a - b for a, b in zip(ps, ex)], mk + "--", color=c, label=f"Psi_G - ED, {dv} Ng={ng}")
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("P_ion from impurity docc (fragments)"); ax[0].set_ylabel("P_ion of normalized Psi_G"); ax[0].legend()
ax[1].axhline(0, color="k", lw=.6); ax[1].set_xlabel("U/t"); ax[1].set_ylabel("P_ion - P_ion(ED)"); ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig("../psi_scan/pion_compare.png", dpi=150)
for k in sorted(rows): print(k, "frag %.4f Psi %.4f ED %.4f norm %.3f" % rows[k])
