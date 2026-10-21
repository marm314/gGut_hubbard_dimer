"""Energy errors of gGut (c-v hopping) vs ED for B_c=1 (fixed-point loop) and B_c=3 (direct minimization).
compute: python3 plot_cederbaum_energies.py compute -> cederbaum_energies.npz  (V in {0.5,1,2}, t_c grid, g=1, E_c=10)
plot:    python3 plot_cederbaum_energies.py         -> cederbaum_energies.png
Panels: (a) E_gGut - E_ED vs t_c for B_c=1 and B_c=3; (b) energy gained by the ghosts, E(B_c=1) - E(B_c=3);
(c) |nu_c - nu_c^ED| for both."""
import os
import sys
import numpy as np

VS = [0.5, 1.0, 2.0]
TCS = [0.5, 0.35, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.025]
FILE, PNG = "cederbaum_data/cederbaum_energies.npz", "cederbaum_data/cederbaum_energies.png"


def compute():
    import gga_cederbaum as g
    import exact_cederbaum as ed
    import not_imp_gga_cederbaum as N
    rng = np.random.default_rng(5)
    out = {}
    for V in VS:
        zprev = None
        for tc in TCS:
            r = ed.solve(1.0, 10.0, V, tc, ht="cv")
            s1 = g.run(1.0, 10.0, V, tc, 1, max_iter=2000, mix=0.5, tol=1e-11)
            m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
            if zprev is None:
                best = None
                for k in range(8):
                    res = N.minimize_once(P, N.random_start(P, rng))
                    if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
                        best = res
            else:
                best = N.minimize_once(P, zprev + 1e-3 * rng.standard_normal(zprev.size))
                if np.abs(P.constraints(best.x)).max() > 1e-9:
                    best = N.minimize_once(P, N.random_start(P, rng))
            zprev = best.x
            o3 = N.recover(P, best.x)
            key = f"V{V}_tc{tc}"
            out[key] = np.array([r["E0"], r["nu_c"], s1["E_var"], s1["nu"], best.fun, o3["nu"],
                                 float(s1["converged"]), np.abs(P.constraints(best.x)).max(), o3["kkt"]])
            print(key, "ED E,nu", r["E0"], r["nu_c"], "| Bc=1 dE %.2e conv %s | Bc=3 dE %.2e cons %.0e kkt %.0e" %
                  (s1["E_var"] - r["E0"], s1["converged"], best.fun - r["E0"], out[key][7], out[key][8]), flush=True)
    np.savez(FILE, **out)


def load_bc5():
    """B_c=5 data from compute_bc5_data.py: key V{V}_tc{tc}_E = [E_ED, nu_ED, E_B3, nu_B3, E_B5, nu_B5, KKT, n_conv, spread, n_rand]."""
    out = {}
    for V in VS:
        f = f"cederbaum_bc5_V{V}.npz"
        if os.path.exists(f):
            out.update({k: v for k, v in np.load(f).items()})
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = np.load(FILE); d5 = load_bc5()
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    colors = {0.5: "tab:blue", 1.0: "tab:green", 2.0: "tab:purple"}
    for V in VS:
        tcs = np.array([t for t in TCS if f"V{V}_tc{t}_E" in d5]); c = colors[V]
        rows = np.array([d[f"V{V}_tc{t}"] for t in TCS]); all_t = np.array(TCS)
        ed_E, ed_nu, e1, n1, e3, n3 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
        axs[0].loglog(all_t, e1 - ed_E, "o-", color=c, lw=1.8)
        axs[0].loglog(all_t, np.maximum(e3 - ed_E, 1e-12), "s--", color=c, lw=1.8, mfc="none")
        axs[1].loglog(all_t, e1 - e3, "s--", color=c, lw=1.8, mfc="none")
        axs[2].loglog(all_t, np.maximum(np.abs(n1 - ed_nu), 1e-12), "o-", color=c, lw=1.8)
        axs[2].loglog(all_t, np.maximum(np.abs(n3 - ed_nu), 1e-12), "s--", color=c, lw=1.8, mfc="none")
        if len(tcs):
            r5 = np.array([d5[f"V{V}_tc{t}_E"] for t in tcs])
            e5, n5 = r5[:, 4], r5[:, 5]
            e1s = np.array([d[f"V{V}_tc{t}"][2] for t in tcs]); eds = r5[:, 0]; nds = r5[:, 1]
            axs[0].loglog(tcs, np.maximum(e5 - eds, 1e-12), "D:", color=c, lw=2.4, ms=5)
            axs[1].loglog(tcs, e1s - e5, "D:", color=c, lw=2.4, ms=5)
            axs[2].loglog(tcs, np.maximum(np.abs(n5 - nds), 1e-12), "D:", color=c, lw=2.4, ms=5)
    for ax in axs:
        ax.set_xlabel(r"$t_c/g$", fontsize=17); ax.tick_params(labelsize=14)
    axs[0].set_ylabel(r"$(E_{\rm gGut}-E_{\rm ED})/g$", fontsize=17)
    axs[1].set_ylabel(r"$(E_{B_c=1}-E_{B_c})/g$", fontsize=17)
    axs[2].set_ylabel(r"$|\nu_c-\nu_c^{\rm ED}|$", fontsize=17)
    vh = [Line2D([0], [0], color=colors[V], lw=2.2, label=rf"$V/g={V:g}$") for V in VS]
    sh = [Line2D([0], [0], color="k", lw=1.8, marker="o", label=r"$B_c=1$"),
          Line2D([0], [0], color="k", lw=1.8, ls="--", marker="s", mfc="none", label=r"$B_c=3$"),
          Line2D([0], [0], color="k", lw=2.4, ls=":", marker="D", ms=5, label=r"$B_c=5$")]
    l1 = axs[0].legend(handles=vh, fontsize=12, loc="upper left")
    axs[0].add_artist(l1)
    axs[0].legend(handles=sh, fontsize=12, loc="center left")
    axs[1].legend(handles=sh[1:], fontsize=12, loc="upper left", title="gain of", title_fontsize=11)
    plt.tight_layout(); plt.savefig(PNG, dpi=150); plt.close()
    print("wrote", PNG)


if __name__ == "__main__":
    compute() if len(sys.argv) > 1 and sys.argv[1] == "compute" else plot()
