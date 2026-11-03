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


def cell(tag, U, key, f):
    d = D[tag]; dv = tag.split("_")[1][:3]
    i = at(d, U); v = f(d, i)
    return v + "$^\\dagger$" if tag.startswith("ng0") and U > Uc[dv] else v


for dv, t2 in (("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n")):
    t0 = f"ng0_{dv}"
    L = ["\\begin{tabular}{r rrrr c rrrr}", "\\toprule",
         " & \\multicolumn{4}{c}{$\\Ng=0$} & & \\multicolumn{4}{c}{$\\Ng=2$}\\\\", "\\cmidrule{2-5}\\cmidrule{7-10}",
         "$U/t$ & $\\langle\\Psi|\\Psi\\rangle$ & $E_{\\rm gGA}-E$ & $E_\\Psi-E$ & $1-|\\langle\\Psi|{\\rm FCI}\\rangle|$ & & "
         "$\\langle\\Psi|\\Psi\\rangle$ & $E_{\\rm gGA}-E$ & $E_\\Psi-E$ & $1-|\\langle\\Psi|{\\rm FCI}\\rangle|$\\\\", "\\midrule"]
    for U in rows:
        r = [f"{U:g}"]
        for tag in (t0, t2):
            r += [cell(tag, U, "", lambda d, i: f"{d['norm'][i]:.3f}"),
                  cell(tag, U, "", lambda d, i: f"{d['EgGA'][i]-d['EFCI'][i]:.3f}"),
                  cell(tag, U, "", lambda d, i: sci(max(d['EPsi'][i] - d['EFCI'][i], 0.0))),
                  cell(tag, U, "", lambda d, i: sci(1 - d['ovl'][i]))]
            if tag == t0: r.append("")
        L.append(" & ".join(r[:5]) + " & & " + " & ".join(r[6:] if len(r) > 9 else r[5:]) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"../doc/tab_psi_energy_{dv}.tex", "w").write("\n".join(L) + "\n")

# ionic weights, both dV
L = ["\\begin{tabular}{r rrrrr c rrrrr}", "\\toprule",
     " & \\multicolumn{5}{c}{$\\dV=0$: $P_{\\rm ion}$} & & \\multicolumn{5}{c}{$\\dV=2$: $P_{\\rm ion}$}\\\\", "\\cmidrule{2-6}\\cmidrule{8-12}",
     "$U/t$ & ED & $\\Psi_{\\Ng=0}$ & $\\Psi_{\\Ng=2}$ & frag$_{\\Ng=0}$ & frag$_{\\Ng=2}$ & & ED & $\\Psi_{\\Ng=0}$ & $\\Psi_{\\Ng=2}$ & frag$_{\\Ng=0}$ & frag$_{\\Ng=2}$\\\\", "\\midrule"]
for U in rows:
    r = [f"{U:g}"]
    for dv, t2 in (("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n")):
        t0 = f"ng0_{dv}"
        r += [f"{D[t2]['Pionex'][at(D[t2], U)]:.4f}",
              cell(t0, U, "", lambda d, i: f"{d['PionPsi'][i]:.4f}"), f"{D[t2]['PionPsi'][at(D[t2], U)]:.4f}",
              cell(t0, U, "", lambda d, i: f"{d['Pionfr'][i]:.4f}"), f"{D[t2]['Pionfr'][at(D[t2], U)]:.4f}"]
        if dv == "dV0": r.append("")
    L.append(" & ".join(r[:6]) + " & & " + " & ".join(r[7:]) + "\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("../doc/tab_psi_ionic.tex", "w").write("\n".join(L) + "\n")

# dV=0 amplitudes and bonding weight (c+s)^2 of the exact cation, Ng=2 and Ng=0
L = ["\\begin{tabular}{r rr c rr c rr c rrr}", "\\toprule",
     " & \\multicolumn{2}{c}{ED} & & \\multicolumn{2}{c}{$\\Psi_{\\Ng=2}$} & & \\multicolumn{2}{c}{$\\Psi_{\\Ng=0}$} & & \\multicolumn{3}{c}{$(\\cos\\theta+\\sin\\theta)^2$}\\\\",
     "\\cmidrule{2-3}\\cmidrule{5-6}\\cmidrule{8-9}\\cmidrule{11-13}",
     "$U/t$ & $\\cos\\theta$ & $\\sin\\theta$ & & $\\cos\\theta$ & $\\sin\\theta$ & & $\\cos\\theta$ & $\\sin\\theta$ & & ED & $\\Psi_{\\Ng=2}$ & $\\Psi_{\\Ng=0}$\\\\", "\\midrule"]
for U in rows:
    d2, d0 = D["ng2_dV0"], D["ng0_dV0"]; i2, i0 = at(d2, U), at(d0, U)
    s2 = np.sign(d2["cov"][i2]); c2, n2 = s2 * d2["cov"][i2], s2 * d2["ion"][i2]
    ce, ne = d2["covex"][i2], d2["ionex"][i2]
    s0 = np.sign(d0["cov"][i0]); c0, n0 = s0 * d0["cov"][i0], s0 * d0["ion"][i0]
    dg = "$^\\dagger$" if U > Uc["dV0"] else ""
    a0 = [f"{c0:.4f}{dg}", f"{n0:.4f}"]; w0 = f"{(c0 + n0) ** 2:.4f}{dg}"
    L.append(" & ".join([f"{U:g}", f"{ce:.4f}", f"{ne:.4f}", "", f"{c2:.4f}", f"{n2:.4f}", ""] + a0 + ["", f"{(ce + ne) ** 2:.4f}", f"{(c2 + n2) ** 2:.4f}", w0]) + "\\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open("../doc/tab_psi_amp.tex", "w").write("\n".join(L) + "\n")

# figure: P_ion and errors
def plot0(a, x, y, uc, log=False, ls="-", **kw):
    """Ng=0 curve: solid up to U_c, dashed above (there Psi_qp is not determined by the solution)."""
    f = a.semilogy if log else a.plot
    lo = x <= uc
    l, = f(x[lo], y[lo], ls, **kw)
    kw.pop("label", None); kw.pop("color", None)
    f(x[~lo | (np.arange(len(x)) == lo.sum() - 1)], y[~lo | (np.arange(len(x)) == lo.sum() - 1)], "--" if ls == "-" else ls, color=l.get_color(), **kw)

fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharex="col")
for j, (dv, t2) in enumerate((("dV0", "ng2_dV0"), ("dV2", "ng2_dV2n"))):
    d0, d2 = D[f"ng0_{dv}"], D[t2]; m0 = d0["U"] <= 10; x2 = d2["U"]
    a = ax[0, j]
    a.plot(d2["U"], d2["Pionex"], "k-", label="ED"); plot0(a, d0["U"][m0], d0["PionPsi"][m0], Uc[dv], color="tab:orange", label=r"$\Psi$, $N_g=0$")
    a.plot(x2, d2["PionPsi"], color="tab:blue", label=r"$\Psi$, $N_g=2$")
    plot0(a, d0["U"][m0], d0["Pionfr"][m0], Uc[dv], ls=":", color="tab:orange", label="fragments, $N_g=0$"); a.plot(x2, d2["Pionfr"], ":", color="tab:blue", label="fragments, $N_g=2$")
    for r_ in range(3): ax[r_, j].axvline(Uc[dv], color="tab:orange", lw=0.8, ls="--")
    a.set_ylabel(r"$P_{\rm ion}$"); a.set_title(rf"$\Delta V={0 if dv=='dV0' else 2}$"); a.set_xlim(0, 10)
    a = ax[1, j]
    plot0(a, d0["U"][m0], np.maximum(d0["EgGA"][m0] - d0["EFCI"][m0], 1e-9), Uc[dv], log=True, ls=":", color="tab:orange", label=r"gGA, $N_g=0$")
    plot0(a, d0["U"][m0], np.maximum(d0["EPsi"][m0] - d0["EFCI"][m0], 1e-9), Uc[dv], log=True, color="tab:orange", label=r"$\Psi$, $N_g=0$")
    a.semilogy(x2, np.maximum(d2["EgGA"] - d2["EFCI"], 1e-9), ":", color="tab:blue", label=r"gGA, $N_g=2$")
    a.semilogy(x2, np.maximum(d2["EPsi"] - d2["EFCI"], 1e-9), color="tab:blue", label=r"$\Psi$, $N_g=2$"); a.set_ylabel(r"$E-E_{\rm FCI}$")
    a = ax[2, j]
    plot0(a, d0["U"][m0], np.maximum(1 - d0["ovl"][m0], 1e-9), Uc[dv], log=True, color="tab:orange"); a.semilogy(x2, np.maximum(1 - d2["ovl"], 1e-9), color="tab:blue")
    a.set_ylabel(r"$1-|\langle\Psi|{\rm FCI}\rangle|$"); a.set_xlabel("$U/t$")
    if j == 0: ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("../psi_scan/psi_summary.png", dpi=130)
print("ok")
