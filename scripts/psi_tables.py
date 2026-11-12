"""LaTeX tables (doc/tab_psi_*.tex) and figure (psi_scan/psi_summary.png) of Sec. 'Gutzwiller wave function' from psi_scan/*.txt (psi_scan.py)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

col = ["U", "norm", "EPsi", "EFCI", "EgGA", "ovl", "PionPsi", "Pionex", "Pionfr", "cov", "ion", "ionas", "kinPsi", "kinex", "kingGA", "wnull", "covex", "ionex", "ionasex"]
D = {}
for tag in ["ng0_dV0", "ng0_dV2", "ng2_dV0", "ng2_dV2n"]:
    a = np.loadtxt(f"../psi_scan/{tag}.txt"); D[tag] = {k: a[:, i] for i, k in enumerate(col)}
Uc = {"dV0": 8.0, "dV2": 9.66}                    # Ng=0 critical U/t: Z=0 above, Psi not determined by the solution
rows = [0.1, 2.0, 4.0, 6.0, 7.0, 8.5, 10.0]
at = lambda d, U: int(np.argmin(np.abs(d["U"] - U)))


def sci(x):
    if x == 0: return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.1f}\\!\\times\\!10^{{{e}}}$"


def cell(tag, U, key, f, psi=True):
    """psi=True: a Psi_G-derived value, not shown ("--") above U_c of Ng=0 (Z=R=0: the solution does not determine Psi_qp)"""
    d = D[tag]; dv = tag.split("_")[1][:3]
    i = at(d, U); v = f(d, i)
    return "--" if psi and tag.startswith("ng0") and U > Uc[dv] else v


# Normalized (Psibar_G) and unnormalized (Psi_G) values side by side: <Psi_G|O|Psi_G> = <Psi_G|Psi_G> <Psibar_G|O|Psibar_G>
# Tables are in two blocks: first all Ng=0 rows, below the Ng=2 rows.
def block(title, ncol, rows_):
    return [f"\\multicolumn{{{ncol}}}{{l}}{{{title}}}\\\\"] + rows_

for dv, t2 in (("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n")):
    hd = ["$\\langle\\Psi_G|\\Psi_G\\rangle$", "$E_{\\rm gGA}-E$", "$E_{\\bar\\Psi_G}-E$", "$\\langle H\\rangle_{\\Psi_G}-E$", "$1-|\\langle\\bar\\Psi_G|{\\rm FCI}\\rangle|$", "$|\\langle\\Psi_G|{\\rm FCI}\\rangle|$"]
    L = ["\\begin{tabular}{r rrrrrr}", "\\toprule", " & ".join(["$U/t$"] + hd) + "\\\\", "\\midrule"]
    for ng, tag in ((0, f"ng0_{dv}"), (2, t2)):
        rr = []
        for U in rows:
            rr.append(" & ".join([f"{U:g}",
                cell(tag, U, "", lambda d, i: f"{d['norm'][i]:.3f}"),
                cell(tag, U, "", lambda d, i: f"{d['EgGA'][i]-d['EFCI'][i]:.3f}", psi=False),
                cell(tag, U, "", lambda d, i: sci(max(d['EPsi'][i] - d['EFCI'][i], 0.0))),
                cell(tag, U, "", lambda d, i: f"{d['norm'][i] * d['EPsi'][i] - d['EFCI'][i]:.3f}"),
                cell(tag, U, "", lambda d, i: sci(1 - d['ovl'][i])),
                cell(tag, U, "", lambda d, i: f"{np.sqrt(d['norm'][i]) * d['ovl'][i]:.4f}")]) + "\\\\")
        L += block(f"$\\Ng={ng}$", 7, rr) + (["\\midrule"] if ng == 0 else [])
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"../doc/tab_psi_energy_{dv}.tex", "w").write("\n".join(L) + "\n")

# ionic weights, both dV side by side; Ng=0 rows first, Ng=2 below
H = ["ED", "$P_{\\rm ion}^{\\bar\\Psi_G}$", "$P_{\\rm ion}^{\\Psi_G}$", "$P_{\\rm ion}^{\\rm loc}$"]
L = ["\\begin{tabular}{r rrrr c rrrr}", "\\toprule",
     " & \\multicolumn{4}{c}{$\\dV=0$} & & \\multicolumn{4}{c}{$\\dV=2$}\\\\", "\\cmidrule{2-5}\\cmidrule{7-10}",
     " & ".join(["$U/t$"] + H + [""] + H) + "\\\\", "\\midrule"]
for ng in (0, 2):
    rr = []
    for U in rows:
        r = [f"{U:g}"]
        for dv, t2 in (("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n")):
            tag = f"ng0_{dv}" if ng == 0 else t2; d2 = D[t2]; i2 = at(d2, U)
            r += [f"{d2['Pionex'][i2]:.4f}",
                  cell(tag, U, "", lambda d, i: f"{d['PionPsi'][i]:.4f}"), cell(tag, U, "", lambda d, i: f"{d['norm'][i] * d['PionPsi'][i]:.4f}"),
                  cell(tag, U, "", lambda d, i: f"{d['Pionfr'][i]:.4f}", psi=False)]
            if dv == "dV0": r.append("")
        rr.append(" & ".join(r) + "\\\\")
    L += block(f"$\\Ng={ng}$", 10, rr) + (["\\midrule"] if ng == 0 else [])
L += ["\\bottomrule", "\\end{tabular}"]
open("../doc/tab_psi_ionic.tex", "w").write("\n".join(L) + "\n")

# dV=0 amplitudes and (c+s)^2 of the exact cation; normalized (bar) and unnormalized amplitudes = sqrt(norm) x normalized; Ng=0 rows first, Ng=2 below
L = ["\\begin{tabular}{r rr c rr rr c rrr}", "\\toprule",
     " & \\multicolumn{2}{c}{ED} & & \\multicolumn{2}{c}{$\\bar\\Psi_G$} & \\multicolumn{2}{c}{$\\Psi_G$} & & \\multicolumn{3}{c}{$(\\cos\\theta+\\sin\\theta)^2$}\\\\",
     "\\cmidrule{2-3}\\cmidrule{5-8}\\cmidrule{10-12}",
     "$U/t$ & $\\cos\\theta$ & $\\sin\\theta$ & & $\\cos\\theta$ & $\\sin\\theta$ & $\\cos\\theta$ & $\\sin\\theta$ & & ED & $\\bar\\Psi_G$ & $\\Psi_G$\\\\", "\\midrule"]
for ng in (0, 2):
    rr = []
    for U in rows:
        d2 = D["ng2_dV0"]; i2 = at(d2, U); ce, ne = d2["covex"][i2], d2["ionex"][i2]
        d = D["ng0_dV0"] if ng == 0 else d2; i = at(d, U)
        sg = np.sign(d["cov"][i]); c, n = sg * d["cov"][i], sg * d["ion"][i]; f_ = np.sqrt(d["norm"][i])
        hide = ng == 0 and U > Uc["dV0"]
        v = ["--"] * 5 if hide else [f"{c:.4f}", f"{n:.4f}", f"{f_ * c:.4f}", f"{f_ * n:.4f}"]
        w = ["--"] * 2 if hide else [f"{(c + n) ** 2:.4f}", f"{f_ ** 2 * (c + n) ** 2:.4f}"]
        rr.append(" & ".join([f"{U:g}", f"{ce:.4f}", f"{ne:.4f}", ""] + v[:4] + ["", f"{(ce + ne) ** 2:.4f}"] + w) + "\\\\")
    L += block(f"$\\Ng={ng}$", 12, rr) + (["\\midrule"] if ng == 0 else [])
L += ["\\bottomrule", "\\end{tabular}"]
open("../doc/tab_psi_amp.tex", "w").write("\n".join(L) + "\n")

# figure: P_ion and errors
def plot0(a, x, y, uc, log=False, ls="-", **kw):
    """Ng=0 curve: solid up to U_c, dashed above (there Psi_qp is not determined by the solution)."""
    f = a.semilogy if log else a.plot
    lo = x <= uc
    l, = f(x[lo], y[lo], ls, **kw)
    if ls in ("-", "-."): return          # Psi_G curves of Ng=0 are not shown above U_c (Psi_qp not determined)
    kw.pop("label", None); kw.pop("color", None)
    f(x[~lo | (np.arange(len(x)) == lo.sum() - 1)], y[~lo | (np.arange(len(x)) == lo.sum() - 1)], "--" if ls == "-" else ls, color=l.get_color(), **kw)

fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharex="col")
for j, (dv, t2) in enumerate((("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n"))):
    d0, d2 = D[f"ng0_{dv}"], D[t2]; m0 = d0["U"] <= 10; x2 = d2["U"]
    a = ax[0, j]
    a.axhline(0, color="0.7", lw=0.8)
    plot0(a, d0["U"][m0], (d0["PionPsi"] - d0["Pionex"])[m0], Uc[dv], color="tab:orange", label=r"$\bar\Psi_G$, $N_g=0$")
    a.plot(x2, d2["PionPsi"] - d2["Pionex"], color="tab:blue", label=r"$\bar\Psi_G$, $N_g=2$")
    plot0(a, d0["U"][m0], (d0["Pionfr"] - d0["Pionex"])[m0], Uc[dv], ls="--", color="tab:orange", label=r"loc, $N_g=0$"); a.plot(x2, d2["Pionfr"] - d2["Pionex"], "--", color="tab:blue", label=r"loc, $N_g=2$")
    plot0(a, d0["U"][m0], (d0["norm"] * d0["PionPsi"] - d0["Pionex"])[m0], Uc[dv], ls="-.", color="tab:orange", label=r"$\Psi_G$ (unnorm.), $N_g=0$"); a.plot(x2, d2["norm"] * d2["PionPsi"] - d2["Pionex"], "-.", color="tab:blue", label=r"$\Psi_G$ (unnorm.), $N_g=2$")
    for r_ in range(3): ax[r_, j].axvline(Uc[dv], color="tab:orange", lw=0.8, ls="--")
    a.set_ylabel(r"$P_{\rm ion}^{X}-P_{\rm ion}^{\rm ED}$"); a.set_title(rf"$\Delta V={0 if dv=='dV0' else 2}$"); a.set_xlim(0, 10)
    if dv == "dV2": a.set_ylim(-0.1, 0.2)      # the unnormalized Ng=2 curve (norm 0.29 at small U/t) goes far below: cut
    a = ax[1, j]
    plot0(a, d0["U"][m0], np.maximum(d0["EgGA"][m0] - d0["EFCI"][m0], 1e-9), Uc[dv], log=True, ls=":", color="tab:orange", label=r"gGA, $N_g=0$")
    plot0(a, d0["U"][m0], np.maximum(d0["EPsi"][m0] - d0["EFCI"][m0], 1e-9), Uc[dv], log=True, color="tab:orange", label=r"$\bar\Psi_G$, $N_g=0$")
    a.semilogy(x2, np.maximum(d2["EgGA"] - d2["EFCI"], 1e-9), ":", color="tab:blue", label=r"gGA, $N_g=2$")
    a.semilogy(x2, np.maximum(d2["EPsi"] - d2["EFCI"], 1e-9), color="tab:blue", label=r"$\bar\Psi_G$, $N_g=2$"); a.set_ylabel(r"$E-E_{\rm FCI}$")
    a = ax[2, j]
    plot0(a, d0["U"][m0], np.maximum(1 - d0["ovl"][m0], 1e-9), Uc[dv], log=True, color="tab:orange"); a.semilogy(x2, np.maximum(1 - d2["ovl"], 1e-9), color="tab:blue")
    a.set_ylabel(r"$1-|\langle\bar\Psi_G|{\rm FCI}\rangle|$"); a.set_xlabel("$U/t$")
from matplotlib.lines import Line2D
hs = [Line2D([], [], color="tab:orange", lw=2, label=r"$N_g=0$"), Line2D([], [], color="tab:blue", lw=2, label=r"$N_g=2$"),
      Line2D([], [], color="0.2", ls="-", label=r"$\bar\Psi_G$ (normalized)"), Line2D([], [], color="0.2", ls="-.", label=r"$\Psi_G$ (unnormalized)"),
      Line2D([], [], color="0.2", ls="--", label=r"loc ($P_{\rm ion}^{\rm loc}$)"), Line2D([], [], color="0.2", ls=":", label=r"gGA ($E_{\rm gGA}$)")]
fig.legend(handles=hs, loc="upper center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("../psi_scan/psi_summary.png", dpi=130)
print("ok")
