"""Spectra of G^h_cc in the units of Golze's Fig. 2 (arXiv:2412.17132): x = (w - eps_QP)/Omega, Omega = 2g, rho = sqrt2 V,
so rho/Omega = V/(sqrt2 g) = 0.707 (V=1) and 1.414 (V=2); rho = sqrt(2) V (Cederbaum 1980, Eq. 28-29). Rows: GW+C (Poisson satellites at eps_QP - n Omega, weights
gamma^n e^-gamma/n!, gamma = rho^2/Omega^2), exact (ED, t_c=0), exact (ED, t_c=0.05, c-v hopping),
gGut B_c=3 and B_c=1 at t_c=0.05 (data from cederbaum_spectra.npz). eps_QP = strongest pole of each row.
python3 plot_golze_units.py -> cederbaum_data/golze_units.png"""
import numpy as np
from math import factorial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exact_cederbaum as ed
import g0w0_cederbaum as gw

import plot_cederbaum_spectra as ps
import plot_cederbaum_energies as pe
D = ps.load_all()
VS = ps.VS
MEAS = np.load("cederbaum_data/golze_fig2_measured.npz")      # peak positions measured in the rendered Fig. 2 of Kocklauner & Golze (measure_golze_fig2.py)
CP = np.load("cederbaum_data/ghost_solution_poles.npz")      # cached B_c=3, 5 solutions (cache_poles.py)
TC, OMEGA, ETA = 0.05, 2.0, 0.018          # ETA in units of Omega: Lorentzian fit to the digitized G0W0 peaks of Golze (0.017-0.019; half-maximum width gives 0.025)
ROWS = [("GW+C", "#0072B2"), ("G$_0$W$_0$", "#56B4E9"), ("exact, $t_c=0$", "#D55E00"), ("exact, $t_c=0.05$", "#D55E00"),
        ("gGut $B_c=1$", "#CC79A7"), ("gGut $B_c=3$", "#009E73"), ("gGut $B_c=5$", "#E69F00")]


def rows(V):
    rho = np.sqrt(2) * V
    gam = (rho / OMEGA) ** 2
    pg, zg = gw.g0w0(rho)
    n = np.arange(6)
    gw_ = (-n.astype(float), gam ** n * np.exp(-gam) / np.array([factorial(k) for k in n]))
    r0 = ed.solve(1.0, 10.0, V, 0.0, ht="cv")
    k = f"V{V}_tc{TC}_"
    out = [gw_, ((pg - pg[0]) / OMEGA, zg), (r0["poles"], r0["weights"]), (D[k + "ed_poles"], D[k + "ed_w"]),
           (D[k + "g1_poles"], D[k + "g1_w"]), (CP[k[:-1] + "_B3_poles"], CP[k[:-1] + "_B3_w"]), (CP[k[:-1] + "_B5_poles"], CP[k[:-1] + "_B5_w"])]
    res = []
    for p, w in out:
        p = np.asarray(p, float); w = np.asarray(w, float); m = w > 1e-4
        p, w = p[m], w[m]
        res.append((p, w))
    return res


def lor(x, p, w):
    return sum(wi * ETA / np.pi / ((x - pi) ** 2 + ETA ** 2) for pi, wi in zip(p, w))


fig, axs = plt.subplots(1, 2, figsize=(11, 7.6), sharex=False)
x = np.linspace(-5.2, 5.6, 6000)
for ax, V in zip(axs, VS):
    kx = MEAS[f"rho{round(np.sqrt(2) * V / OMEGA, 1)}_g0w0_curve_x"]
    XL = (kx.min(), kx.max())                       # same x range as the frame of Golze's Fig. 2b,c
    ax.set_xlim(*XL)
    for i, ((p, w), (lab, col)) in enumerate(zip(rows(V), ROWS)):
        if i < 2:
            pu = p                                   # already in units of Omega, QP at 0
        else:
            pu = (p - p[np.argmax(w)]) / OMEGA       # gGut/ED poles: QP = strongest pole
        off = 1.35 * (len(ROWS) - 1 - i)
        y = lor(x, pu, w)
        y = y / y.max() if y.max() > 0 else y
        ax.plot(x, off + y, color=col, lw=1.4)
        ax.vlines(pu, off, off + w / w.max(), color=col, lw=1, alpha=0.5)
        ax.axhline(off, color="0.85", lw=0.6, zorder=0)
        if lab.startswith("G$_0$W$_0$"):                       # reference G0W0 positions of Kocklauner & Golze (measured from their Fig. 2)
            kk = f"rho{round(np.sqrt(2) * V / OMEGA, 1)}_g0w0_curve_"
            xc, yc = MEAS[kk + "x"], MEAS[kk + "y"]
            ax.plot(xc, off + yc / yc.max(), "--", color="0.15", lw=1.1, zorder=5)      # digitized curve of Kocklauner & Golze
        ax.text(XL[0] + 0.1, off + 0.55, lab, ha="left", fontsize=9, color=col)
    r0 = ed.solve(1.0, 10.0, V, 0.0, ht="cv")                     # vertical lines at the exact pole positions (t_c=0)
    for xe in (r0["poles"][r0["weights"] > 1e-3] - r0["poles"][0]) / OMEGA:
        ax.axvline(xe, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_title(rf"$\rho/\Omega={np.sqrt(2)*V/OMEGA:.1f}$  ($V/g={V:.3f}$)")
    ax.set_xlabel(r"$(\omega-\epsilon_{QP})/\Omega$"); ax.set_yticks([])
axs[0].set_ylabel("intensity (each row normalized to its tallest peak)")
fig.suptitle(r"$G^h_{cc}$ in the units of Golze's Fig. 2 ($\Omega=2g$)", y=0.995)
fig.text(0.5, 0.004, "dashed: G$_0$W$_0$ curve of Kockl\u00e4uner and Golze (digitized from their Fig. 2, normalized to its maximum)", ha="center", fontsize=9)
fig.subplots_adjust(left=0.055, right=0.995, top=0.91, bottom=0.105, wspace=0.03)
fig.savefig("cederbaum_data/golze_units.png", dpi=160)
for V in VS:
    print("V", V)
    for (p, w), (lab, _) in zip(rows(V), ROWS):
        pu = p if lab in ("GW+C", "G$_0$W$_0$") else (p - p[np.argmax(w)]) / OMEGA
        print(f"  {lab:20s}", np.round(pu, 3), np.round(w / (w.sum() if lab != 'GW+C' else 1), 3))
