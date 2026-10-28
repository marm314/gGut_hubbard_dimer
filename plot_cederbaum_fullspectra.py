"""Full spectral function A = A_cc + A_oo + A_vv (hole and particle sides), exact (ED of H + H_t^cv) vs gGut B_c=3, 5
(cached direct-minimization solutions, ghost_solutions.npz), E_c=10, g=1, t_c = 0.1, 0.05, rho/Omega = 0.5, 1.0.
gGut poles: all eigenvalues of H_qp (occupied = hole, empty = particle), weight on orbital I = (R_I^T phi_n[I])^2, per spin.
ED: <n|c^dag_I|GS>^2 (particle, omega = E_n-E_0) and <n|c_I|GS>^2 (hole, omega = -(E_n-E_0)) for I = c, o, v (spin up).
Usage: python3 plot_cederbaum_fullspectra.py   -> cederbaum_data/cederbaum_fullspectra.png"""
import numpy as np

ETA = 0.15
VS = [0.7071067811865476, 1.4142135623730951]
TCS = [0.1, 0.05]
PNG = "cederbaum_data/cederbaum_fullspectra.png"
ORB = ("c", "o", "v")


def ed_poles(V, tc):
    import exact_cederbaum as ed
    H, c, n = ed.build_hamiltonian(1.0, 10.0, V, tc, ht="cv")
    E, U = np.linalg.eigh(H)
    gs = U[:, 0]
    out = []
    for idx in (ed.C_UP, ed.O_UP, ed.V_UP):
        wh = (U.T @ (c[idx] @ gs)) ** 2
        wp = (U.T @ (c[idx].T @ gs)) ** 2
        p = np.concatenate([-(E - E[0]), E - E[0]])
        w = np.concatenate([wh, wp])
        k = w > 1e-9
        out.append((p[k], w[k]))
    return out                                    # [(poles, weights)] for c, o, v


def gut_poles(V, tc, Bc, z):
    import gga_cederbaum as g
    import not_imp_gga_cederbaum as N
    m = g.Model(1.0, 10.0, V, tc, Bc, 1e-9, "cv"); P = N.Problem(m)
    o = N.recover(P, z)
    eps, phi, R = o["eps"], np.linalg.eigh(o["Hqp"])[1], o["R"]
    W = np.array([[(R[I] @ phi[m.sl[I], k]) ** 2 for k in range(eps.size)] for I in range(3)])
    # hole side: occupied levels (weight W); particle side: empty levels (weight W)
    return eps, W, m.Nocc, o["comm"]


def lorentz(w, poles, wts):
    return sum(wt * (ETA / np.pi) / ((w - p) ** 2 + ETA ** 2) for p, wt in zip(poles, wts))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.load("cederbaum_data/ghost_solutions.npz")
    fig, axs = plt.subplots(len(TCS), len(VS), figsize=(13, 7.4), sharex=True)
    w = np.linspace(-18.0, 14.0, 9000)
    col = {"c": "tab:red", "o": "tab:blue", "v": "tab:green"}
    for i, tc in enumerate(TCS):
        for j, V in enumerate(VS):
            ax = axs[i, j]
            ed = ed_poles(V, tc)
            ep = np.concatenate([p for p, _ in ed]); ew = np.concatenate([x for _, x in ed])
            ax.plot(w, lorentz(w, ep, ew), "-", lw=2.2, color="black", label="exact (ED)")
            for Bc, ls, lc, lw in ((3, "--", "red", 2.0), (5, ":", "tab:green", 3.0)):
                eps, W, nocc, comm = gut_poles(V, tc, Bc, Z[f"V{V}_tc{tc}_z{Bc}"])
                ax.plot(w, lorentz(w, eps, W.sum(axis=0)), ls, lw=lw, color=lc, label=rf"gGut ($B_c={Bc}$)")
                print(f"V={V:.3f} tc={tc} Bc={Bc} comm={comm:.1e}  poles", np.round(eps, 3), " weights/orbital", np.round(W, 5).tolist(), flush=True)
            # ED sticks (markers) of weight > 1e-3 coloured by orbital
            for (p, x), o in zip(ed, ORB):
                k = x > 1e-3
                ax.plot(p[k], np.full(k.sum(), -0.12), "o", ms=7, color=col[o], mec="none", label=f"ED peak ({o})" if (i, j) == (0, 0) else None)
            ax.axvline(0, color="0.7", lw=0.8)
            ax.set_xlim(-18.0, 14.0); ax.set_ylim(-0.25, 2.6)
            ax.tick_params(labelsize=14)
            if i == len(TCS) - 1:
                ax.set_xlabel(r"$\omega/g$", fontsize=17)
            if j == 0:
                ax.set_ylabel(r"$g\,A(\omega)$", fontsize=17)
            ax.text(0.03, 0.93, rf"$\rho/\Omega={np.sqrt(2)*V/2:.1f},\ t_c/g={tc:g}$", transform=ax.transAxes, fontsize=14, va="top")
            if (i, j) == (0, 0):
                ax.legend(fontsize=11, loc="center right")
    plt.tight_layout()
    plt.savefig(PNG, dpi=150); plt.close()
    print("wrote", PNG)


if __name__ == "__main__":
    main()
