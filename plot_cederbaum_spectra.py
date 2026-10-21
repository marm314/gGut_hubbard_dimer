"""Hole spectral function of the core orbital, exact (ED of H+H_t^cv) vs gGut (B_c=3, direct minimization,
Eq. GFgGut), for t_c = 0.1 and 0.05 and V = 0.5, 1, 2 (g=1, E_c=10).
Step 1 (compute): python3 plot_cederbaum_spectra.py compute   -> cederbaum_spectra.npz
Step 1b: python3 plot_cederbaum_spectra.py addbc1 (adds the B_c=1 loop spectra)
Step 2 (plot):    python3 plot_cederbaum_spectra.py           -> cederbaum_spectra.png
A^h(w) = sum_n w_n (eta/pi) / ((w - w_n)^2 + eta^2), per spin, Lorentzian eta = ETA."""
import os
import sys
import numpy as np

ETA = 0.15
VS = [0.5, 1.0, 2.0]
TCS = [0.1, 0.05]
FILE = "cederbaum_data/cederbaum_spectra.npz"
PNG = "cederbaum_data/cederbaum_spectra.png"


def compute():
    import gga_cederbaum as g
    import exact_cederbaum as ed
    import not_imp_gga_cederbaum as N
    data = {}
    rng = np.random.default_rng(5)
    for V in VS:
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
    np.savez(FILE, **data)


def add_bc1():
    """Add the B_c=1 gGut poles/weights (fixed-point loop; converges) to the saved data."""
    import gga_cederbaum as g
    d = dict(np.load(FILE))
    for V in VS:
        for tc in TCS:
            sol = g.run(1.0, 10.0, V, tc, 1, max_iter=2000, mix=0.5, tol=1e-11)
            key = f"V{V}_tc{tc}"
            d[key + "_g1_poles"] = sol["poles"]; d[key + "_g1_w"] = sol["weights"]
            print(key, "B_c=1 loop conv", sol["converged"], "poles", np.round(sol["poles"], 4), "w", np.round(sol["weights"], 5))
    np.savez(FILE, **d)


def load_bc5():
    out = {}
    for V in VS:
        f = f"cederbaum_bc5_V{V}.npz"
        if os.path.exists(f):
            out.update({k: v for k, v in np.load(f).items()})
    return out


def lorentz(w, poles, wts):
    return sum(wt * (ETA / np.pi) / ((w - p) ** 2 + ETA ** 2) for p, wt in zip(poles, wts))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(FILE); d5 = load_bc5()
    fig, axs = plt.subplots(len(TCS), len(VS), figsize=(13, 7.2), sharex=True)
    w = np.linspace(-18.0, -5.5, 2500)
    for i, tc in enumerate(TCS):
        for j, V in enumerate(VS):
            ax = axs[i, j]; key = f"V{V}_tc{tc}"
            ed_p, ed_w = d[key + "_ed_poles"], d[key + "_ed_w"]
            gg_p, gg_w = d[key + "_gg_poles"], d[key + "_gg_w"]
            keep = ed_w > 1e-6
            ax.plot(w, lorentz(w, ed_p[keep], ed_w[keep]), "-", lw=2.2, color="black", label="exact")
            ax.plot(w, lorentz(w, d[key + "_g1_poles"], d[key + "_g1_w"]), ":", lw=2.6, color="tab:blue", label="gGut ($B_c=1$)")
            ax.plot(w, lorentz(w, gg_p, gg_w), "--", lw=2.2, color="red", label="gGut ($B_c=3$)")
            k5 = f"V{V}_tc{tc}"
            if k5 + "_poles" in d5:
                ax.plot(w, lorentz(w, d5[k5 + "_poles"], d5[k5 + "_w"]), ":", lw=3.2, color="tab:green", label="gGut ($B_c=5$)")
            ax.set_xlim(-18.0, -5.5)
            ax.tick_params(labelsize=14)
            if i == len(TCS) - 1:
                ax.set_xlabel(r"$\omega/g$", fontsize=17)
            if j == 0:
                ax.set_ylabel(r"$g\,A^h_{cc}(\omega)$", fontsize=17)
            ax.text(0.04, 0.92, rf"$V/g={V:g},\ t_c/g={tc:g}$", transform=ax.transAxes, fontsize=14, va="top")
            if i == 0 and j == 0:
                ax.legend(fontsize=14, loc="center left")
    plt.tight_layout()
    plt.savefig(PNG, dpi=150); plt.close()
    print("wrote", PNG)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute()
    elif len(sys.argv) > 1 and sys.argv[1] == "addbc1":
        add_bc1()
    else:
        plot()
