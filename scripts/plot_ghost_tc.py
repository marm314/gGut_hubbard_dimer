"""Plot poles of G^h_cc (occupied H_qp levels) vs t_c/g from cederbaum_data/scan_ghost_tc/V*.npz. Filled = QP pole (largest weight),
open = zero-weight (muted) ghost poles, grey line = exact poles with weight>1e-3. /usr/bin/python3 scripts/plot_ghost_tc.py"""
import os, numpy as np, re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Vs = [0.7071067811865476, 1.4142135623730951]; fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
for j, V in enumerate(Vs):
    if not os.path.exists(f"cederbaum_data/scan_ghost_tc/V{V}.npz"): continue
    d = np.load(f"cederbaum_data/scan_ghost_tc/V{V}.npz"); tcs = sorted({float(re.match(r"tc([\d.]+)_", k).group(1)) for k in d.files})
    for i, B in enumerate((3, 5)):
        a = ax[i, j]
        for tc in tcs:
            k = f"tc{tc}_"
            if k + "ed_poles" in d.files:
                p, w = d[k + "ed_poles"], d[k + "ed_w"]; a.plot([tc] * (w > 1e-3).sum(), p[w > 1e-3], "_", c="grey", ms=10)
            if k + f"B{B}_poles" not in d.files: continue
            p, w = d[k + f"B{B}_poles"], d[k + f"B{B}_w"]; q = np.argmax(w)
            a.plot(tc, p[q], "o", c="C0", ms=4, label="QP pole" if tc == tcs[0] else None)
            g = [x for x in range(len(p)) if x != q]
            a.plot([tc] * len(g), p[g], "o", mfc="none", c="C3", ms=5, label="ghost / muted (w<1e-2)" if tc == tcs[0] else None)
        a.set_title(f"B_c={B}, V/g={V:.4f} (Golze)"); a.set_ylim(-17, 2)
        if i == 1: a.set_xlabel("$t_c/g$")
        if j == 0: a.set_ylabel("pole / g")
fig.suptitle("Occupied H_qp levels (poles of $G^h_{cc}$); grey = exact satellites (w>1e-3)"); from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],marker="o",ls="",c="C0",ms=5,label="QP pole"),Line2D([],[],marker="o",ls="",mfc="none",c="C3",ms=6,label="ghost / muted pole (min-norm multipliers)"),Line2D([],[],marker="_",ls="",c="grey",ms=10,label="exact satellites (w>1e-3)")],loc="lower center",ncol=3,frameon=False)
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig("cederbaum_data/ghost_poles_vs_tc.png", dpi=130)
