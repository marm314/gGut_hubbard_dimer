"""Figure for doc/cederbaum_gauge.tex: hole poles of G^h_cc vs t_c/g after placing the weightless ghost poles on the ED satellites (data cederbaum_data/ghost_placed/V*.npz); in the B_c=5 panels the free empty ghost level (triangles) is placed on the ED particle-side pole (grey).
Red open circles = placed ghost poles; small black dots = ghost poles in the min-norm gauge; blue = QP pole; grey bars = ED poles (weight > 1e-3). /usr/bin/python3 scripts/plot_ghost_placed.py"""
import re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
Vs = [0.7071067811865476, 1.4142135623730951]; fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
for j, V in enumerate(Vs):
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz"); s = np.load(f"cederbaum_data/scan_ghost_tc_eps1e-07/V{V}.npz")
    for i, B in enumerate((3, 5)):
        a = ax[i, j]; T = sorted(float(re.match(r"tc([\d.]+)_", k).group(1)) for k in d.files if k.endswith(f"B{B}_targets"))
        for t in T:
            p, w = s[f"tc{t}_ed_poles"], s[f"tc{t}_ed_w"]; a.plot([t] * (w > 1e-3).sum(), p[w > 1e-3], "_", c="grey", ms=9)
            k = f"tc{t}_B{B}_"; a.plot([t] * len(d[k + "ghost_before"]), d[k + "ghost_before"], ".", c="k", ms=3); a.plot([t] * len(d[k + "ghost_after"]), d[k + "ghost_after"], "o", mfc="none", c="C3", ms=6)
            a.plot(t, d[k + "qp_after"], "o", c="C0", ms=3)
            if B == 5: a.plot(t, d[k + "empty_target"], "_", c="grey", ms=9); a.plot(t, d[k + "empty_before"], ".", c="k", ms=3); a.plot(t, d[k + "empty_after"], "^", mfc="none", c="C3", ms=6)
        a.set_ylim(-20, 8); a.set_title(f"$B_c={B}$, $V/g={V:.3f}$")
        if i == 1: a.set_xlabel("$t_c/g$")
        if j == 0: a.set_ylabel("pole / $g$")
fig.legend(handles=[Line2D([], [], marker="o", c="C0", ls="", ms=4, label="QP pole"), Line2D([], [], marker="o", mfc="none", c="C3", ls="", ms=6, label="ghost poles, placed on ED satellites"),
    Line2D([], [], marker=".", c="k", ls="", ms=5, label="ghost poles, min-norm gauge"), Line2D([], [], marker="_", c="grey", ls="", ms=9, label="ED poles ($w>10^{-3}$; $B_c=5$ top: particle-side target)"), Line2D([], [], marker="^", mfc="none", c="C3", ls="", ms=6, label="free empty ghost ($B_c=5$), placed")], loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("cederbaum_data/ghost_placed_vs_tc.png", dpi=150)
