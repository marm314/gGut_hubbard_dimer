"""Ionic weight P_ion = <n_0up n_0dn> + <n_1up n_1dn> of the Hubbard-dimer ground state vs U/t: ED against gGA (Ng=2 and Ng=4), for Delta V = 0 and 2.
compute: python3 scripts/plot_ionic_weight.py compute  -> merged_Ng/ionic_weight_data.npz   (one logged line per U/t)
plot:    python3 scripts/plot_ionic_weight.py          -> merged_Ng/ionic_weight_vs_U.png"""
import re
import sys
import numpy as np
import ionic_covalent as ic

FILE, PNG = "merged_Ng/ionic_weight_data.npz", "merged_Ng/ionic_weight_vs_U.png"
Us = np.round(np.arange(0.1, 10.01, 0.1), 1)
UC = {0: 6.56, 2: 8.34}   # critical U/t of the spectral-gap closing (doc/results.tex)


def dirs(dV, ng):
    if dV == 0:
        if ng == 2:
            br = {}
            for line in open("scan_ng2_not_imp/lowest_energy.txt"):
                if not line.startswith("#") and line.strip():
                    f = line.split(); br[round(float(f[0]), 1)] = f[2]
            return lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}"
        return lambda U: f"scan_ng4_not_imp/U_{U:.1f}"
    return lambda U: f"scan_ng{ng}_dV2_confirmed/U_{U:.1f}"


def header(d):
    h = open(d + "/output.out").read()
    return float(re.search(r"E_var/t=([\d.eE+-]+)", h).group(1)), re.search(r"docc=([\d.eE+-]+)", h)


def compute():
    """dV=0: 2*docc from the solver output (the embedding states of the solution). dV=2: the solver output has no docc, so P_ion is recomputed from the impurity
    ground state at the converged (R, lambda); a point is kept only if that reproduces the E_var of the solution (|dE| < 1e-4), otherwise (degenerate ghost
    levels: the embedding ground state is not the state of the solution) it is left out (NaN)."""
    out = {"U": Us}
    for dV in (0, 2):
        out[f"ed_dV{dV}"] = np.array([ic.ed_weights(U, 2.0 * dV)[0] for U in Us])
        for ng in (2, 4):
            gd = dirs(dV, ng); vals = []
            for U in Us:
                Eh, md = header(gd(U))
                if dV == 0:
                    v = 2.0 * float(md.group(1)); note = "solver docc"
                else:
                    v, _, _, _ = ic.gga_weights(gd(U), U, float(dV)); dE = abs(ic.gga_weights.E_var - Eh)
                    note = f"dE={dE:.1e}"
                    if dE > 1e-4: v = np.nan; note += "  -> left out"
                vals.append(v)
                print(f"dV={dV} Ng={ng} U/t={U:.1f}  P_ion gGA={v:.4f} ({note})", flush=True)
            out[f"gga_dV{dV}_ng{ng}"] = np.array(vals)
    np.savez(FILE, **out)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(FILE); U = d["U"]
    C2, C4 = "#0072B2", "#D55E00"                      # categorical: Ng=2, Ng=4 (validated, CVD-safe); ED in neutral ink
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, dV in zip(axs, (0, 2)):
        ax.plot(U, d[f"ed_dV{dV}"], color="0.1", lw=2.2, label="ED (exact)")
        ax.plot(U, d[f"gga_dV{dV}_ng2"], color=C2, lw=2.0, label=r"gGA, $N_g=2$")
        ax.plot(U, d[f"gga_dV{dV}_ng4"], color=C4, lw=2.0, ls=(0, (4, 3)), label=r"gGA, $N_g=4$")
        ax.axvline(UC[dV], color="0.55", lw=1.0, ls=":")
        ax.text(UC[dV] + 0.15, 0.93 if dV == 2 else 0.5, rf"$U_c/t={UC[dV]:.2f}$", color="0.35", fontsize=11, va="top")
        ax.set_xlabel(r"$U/t$"); ax.set_title(rf"$\Delta V={dV}$", fontsize=13)
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, loc="upper right" if dV == 0 else "center right")
    axs[0].set_ylabel(r"ionic weight $P_{\rm ion}=\langle n_{0\uparrow}n_{0\downarrow}\rangle+\langle n_{1\uparrow}n_{1\downarrow}\rangle$")
    plt.tight_layout(); plt.savefig(PNG, dpi=150); plt.close(fig); print("wrote", PNG)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute()
    else:
        plot()
