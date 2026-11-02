"""Intervals in which the weightless ghost poles can be moved (gauge) vs t_c/g, from cederbaum_data/scan_ghost_tc/range_V*.npz.
Bar = sampled range over ACCEPTED shifts only: ghost still identified, QP pole/weights unchanged, [H_qp,rho] unchanged, every level's occupation as close to 0/1 as at the min-norm point.
/usr/bin/python3 scripts/plot_ghost_range.py"""
import os, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
EPS = float(os.environ.get('GHOST_EPS', '1e-9')); SUF = '' if EPS == 1e-9 else f'_eps{EPS:g}'
MODE = os.environ.get('GHOST_MODE', 'auf'); KS = '_auf' if MODE == 'auf' else ''
Vs = [0.7071067811865476, 1.4142135623730951]; fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True); dmax = {}
for j, V in enumerate(Vs):
    r = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/range_V{V}.npz"); s = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
    tcs = sorted({float(re.match(r"tc([\d.]+)_", k).group(1)) for k in r.files})
    for i, B in enumerate((3, 5)):
        a = ax[i, j]; T = [t for t in tcs if f"tc{t}_B{B}_lo" in r.files]
        for q in range(1 if B == 3 else 2):
            for t in T:
                lo, hi = r[f"tc{t}_B{B}_lo{KS}"], r[f"tc{t}_B{B}_hi{KS}"]
                if len(lo) > q: a.plot([t, t], [max(lo[q], -19.9), min(hi[q], 3.9)], c=f"C{3 + q}", lw=4, alpha=0.5, solid_capstyle="butt")
                if len(lo) > q: a.plot(t, r[f"tc{t}_B{B}_min_norm"][q], "o", mfc="none", c="k", ms=3)
        for t in T:
            p, w = s[f"tc{t}_ed_poles"], s[f"tc{t}_ed_w"]; a.plot([t] * (w > 1e-3).sum(), p[w > 1e-3], "_", c="grey", ms=8)
        a.plot(T, [r[f"tc{t}_B{B}_qp"] for t in T], "o", c="C0", ms=3)
        for q in r.files[:0]: pass
        for q in ("occ", "qp", "wq", "wt", "nong", "comm"): dmax[q] = max(dmax.get(q, 0), max(r[f"tc{t}_B{B}_dev_{q}"] for t in T))
        a.set_ylim(-20, 4); a.set_title(f"B_c={B}, V/g={V:.4f} (Golze)")
        if i == 1: a.set_xlabel("$t_c/g$")
        if j == 0: a.set_ylabel("pole / g")
fig.legend(handles=[Line2D([], [], c="C3", lw=4, alpha=0.5, label="reachable range, ghost 1 (clipped at axis)"), Line2D([], [], c="C4", lw=4, alpha=0.5, label="reachable range, ghost 2 (B_c=5)"),
    Line2D([], [], marker="o", mfc="none", c="k", ls="", ms=4, label="min-norm choice"), Line2D([], [], marker="o", c="C0", ls="", ms=4, label="QP pole (unchanged)"),
    Line2D([], [], marker="_", c="grey", ls="", ms=9, label="exact satellites")], loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle(f"Gauge freedom of the weightless ghost poles (occupations kept 0/1{", aufbau imposed" if KS else ""}), shift eps={EPS:g}"); fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(("cederbaum_data/ghost_range_vs_tc.png" if KS else "cederbaum_data/ghost_range_noaufbau_vs_tc.png"), dpi=130)
print("max deviation over all points/accepted shifts:", {k: f"{v:.1e}" for k, v in dmax.items()})
