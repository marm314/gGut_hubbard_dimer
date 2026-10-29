"""Hole spectral function of the core orbital, exact (ED of H+H_t^cv) vs gGut (B_c=3, direct minimization,
Eq. GFgGut), for t_c = 0.1 and 0.05 and V = 0.5, 1, 2 (g=1, E_c=10).
Step 1 (compute): python3 scripts/plot_cederbaum_spectra.py compute   -> cederbaum_spectra.npz
Step 1b: python3 scripts/plot_cederbaum_spectra.py addbc1 (adds the B_c=1 loop spectra)
Step 2 (plot):    python3 scripts/plot_cederbaum_spectra.py           -> cederbaum_spectra.png
A^h(w) = sum_n w_n (eta/pi) / ((w - w_n)^2 + eta^2), per spin, Lorentzian eta = ETA."""
import os
import sys
import numpy as np

ETA = 0.15
VS = [0.7071067811865476, 1.4142135623730951]       # V/g = 1/sqrt2, sqrt2: rho/Omega = V/(sqrt2 g) = 0.5, 1.0 (Golze ratios)
TCS = [0.1, 0.05]
FILE = "cederbaum_data/cederbaum_spectra.npz"


def fileV(V):
    return f"cederbaum_data/cederbaum_spectra_V{V}.npz"

PNG = "cederbaum_data/cederbaum_spectra.png"


def load_all():
    d = {}
    for V in VS:
        d.update(dict(np.load(fileV(V))))
    return d


def compute(Vsel):
    import gga_cederbaum as g
    import exact_cederbaum as ed
    import not_imp_gga_cederbaum as N
    data = {}
    rng = np.random.default_rng(5)
    for V in [Vsel]:
        zprev = None
        for tc in TCS:
            m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
            r = ed.solve(1.0, 10.0, V, tc, ht="cv")
            if zprev is None:
                best = None
                for k in range(8):
                    res = N.minimize_once(P, N.random_start(P, rng))
                    if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
                        best = res
            else:
                best = N.minimize_once(P, zprev + 1e-3 * rng.standard_normal(zprev.size))
            zprev = best.x
            out = N.recover(P, best.x)
            key = f"V{V}_tc{tc}"
            data[key + "_ed_poles"] = r["poles"]; data[key + "_ed_w"] = r["weights"]
            data[key + "_gg_poles"] = out["poles"]; data[key + "_gg_w"] = out["weights"]
            data[key + "_E"] = np.array([best.fun, r["E0"]])
            print(key, "E gGut/ED", best.fun, r["E0"], "gGut poles", np.round(out["poles"], 3), "w", np.round(out["weights"], 4), flush=True)
    np.savez(fileV(Vsel), **data)


def add_bc1(Vsel):
    """Add the B_c=1 gGut poles/weights (fixed-point loop; converges) to the saved data."""
    import gga_cederbaum as g
    d = dict(np.load(fileV(Vsel)))
    for V in [Vsel]:
        for tc in TCS:
            sol = g.run(1.0, 10.0, V, tc, 1, max_iter=2000, mix=0.5, tol=1e-11)
            key = f"V{V}_tc{tc}"
            d[key + "_g1_poles"] = sol["poles"]; d[key + "_g1_w"] = sol["weights"]
            print(key, "B_c=1 loop conv", sol["converged"], "poles", np.round(sol["poles"], 4), "w", np.round(sol["weights"], 5))
    np.savez(fileV(Vsel), **d)


def load_bc5():
    out = {}
    for V in VS:
        f = f"cederbaum_data/cederbaum_bc5_V{V}.npz"
        if os.path.exists(f):
            out.update({k: v for k, v in np.load(f).items()})
    return out


def lorentz(w, poles, wts):
    return sum(wt * (ETA / np.pi) / ((w - p) ** 2 + ETA ** 2) for p, wt in zip(poles, wts))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = load_all(); d5 = load_bc5()
    CP = np.load("cederbaum_data/ghost_solution_poles.npz")      # poles/weights of the cached B_c=3, 5 solutions (cache_poles.py)
    GL = np.load("cederbaum_data/golze_ghost_levels.npz")       # ghost poles as returned by the minimizer (ghost_levels_golze.py, *_orig)
    fig, axs = plt.subplots(len(TCS), len(VS), figsize=(13, 7.2), sharex=True)
    w = np.linspace(-18.0, 0.8, 3500)
    for i, tc in enumerate(TCS):
        for j, V in enumerate(VS):
            ax = axs[i, j]; key = f"V{V}_tc{tc}"
            ed_p, ed_w = d[key + "_ed_poles"], d[key + "_ed_w"]
            gg_p, gg_w = CP[key + "_B3_poles"], CP[key + "_B3_w"]
            keep = ed_w > 1e-6
            ax.plot(w, lorentz(w, ed_p[keep], ed_w[keep]), "-", lw=2.2, color="black", label="exact")
            ax.plot(w, lorentz(w, d[key + "_g1_poles"], d[key + "_g1_w"]), ":", lw=2.6, color="tab:blue", label="gGut ($B_c=1$)")
            ax.plot(w, lorentz(w, gg_p, gg_w), "--", lw=2.2, color="red", label="gGut ($B_c=3$)")
            k5 = f"V{V}_tc{tc}"
            ax.plot(w, lorentz(w, CP[k5 + "_B5_poles"], CP[k5 + "_B5_w"]), ":", lw=3.2, color="tab:green", label="gGut ($B_c=5$)")
            # zero-weight ghost poles (occupied ghost levels of H_qp, as returned by the minimizer) as muted dots below zero
            for Bc, pk, wk, mc, msz in ((3, gg_p, gg_w, "#e8a0a0", 7), (5, CP[key + "_B5_poles"], CP[key + "_B5_w"], "#a6d6a6", 13)):
                if pk is None: continue
                rel = GL[f"V{V}_tc{tc}_B{Bc}_ghost_rel_orig"]
                ax.plot(pk[np.argmax(wk)] + 2.0 * rel, -0.13 + 0 * rel, "o", color=mc, ms=msz, alpha=0.95, mec="none", zorder=1 if Bc == 5 else 2)
            ax.set_xlim(-18.0, 0.8)
            ax.set_ylim(bottom=-0.26)
            ax.tick_params(labelsize=14)
            if i == len(TCS) - 1:
                ax.set_xlabel(r"$\omega/g$", fontsize=17)
            if j == 0:
                ax.set_ylabel(r"$g\,A^h_{cc}(\omega)$", fontsize=17)
            ax.text(0.04, 0.92, rf"$\rho/\Omega={np.sqrt(2)*V/2:.1f},\ t_c/g={tc:g}$", transform=ax.transAxes, fontsize=14, va="top")
            if i == 0 and j == 0:
                from matplotlib.lines import Line2D
                hh, ll = ax.get_legend_handles_labels()
                hh += [Line2D([0], [0], marker="o", ls="", color="#e8a0a0", ms=7, mec="none"), Line2D([0], [0], marker="o", ls="", color="#a6d6a6", ms=11, mec="none")]
                ll += ["ghost poles ($B_c=3$)", "ghost poles ($B_c=5$)"]
                ax.legend(hh, ll, fontsize=12, loc="center left")
    plt.tight_layout()
    plt.savefig(PNG, dpi=150); plt.close()
    print("wrote", PNG)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute(float(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "addbc1":
        add_bc1(float(sys.argv[2]))
    else:
        plot()
