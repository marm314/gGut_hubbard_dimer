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
MROW = {3: -0.10, 5: -0.26, "ov": -0.52}                      # y of the rows of muted poles
MS = {3: 8, 5: 8, "ov": 8}
MC = {3: "#e8a0a0", 5: "#a6d6a6", "ov": "#c3a6e0"}


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


def merged_peaks(ed, wmin=1e-3, tol=1e-3):
    """ED lines of all orbitals (hole and particle) closer than tol merged: list of (omega, weight, {orbital: weight}), as in CederbaumNg0_numerics.py."""
    P, W, O = [], [], []
    for (p, x), o in zip(ed, ORB):
        P += list(p); W += list(x); O += [o] * len(p)
    groups = []
    for i in np.argsort(P):
        if groups and abs(P[i] - groups[-1][0] / groups[-1][1]) < tol:
            gr = groups[-1]; gr[0] += P[i] * W[i]; gr[1] += W[i]; gr[2][O[i]] = gr[2].get(O[i], 0.0) + W[i]
        else:
            groups.append([P[i] * W[i], W[i], {O[i]: W[i]}])
    return [(g[0] / g[1], g[1], g[2]) for g in groups if g[1] >= wmin]


def label_ed(ax, ed, lo, hi, fs=11):
    """Label the ED peaks as in CederbaumNg0.tex: eps_QP^I (main peak of the orbital I = c, o, v) and eps_s1^c, eps_s2^c, ...
    (core satellites, numbered by increasing distance from the main core peak)."""
    pk = merged_peaks(ed); h = lambda q: q[1] / (np.pi * ETA)
    for p, x, orb in pk:
        m = max(orb, key=orb.get)
        if m in ("o", "v") and x > 0.5 and lo <= p <= hi:
            ax.annotate(rf"$\varepsilon_{{\rm QP}}^{{{m}}}$", (p, h((p, x))), xytext=(-3 if m == "o" else 3, 5), textcoords="offset points",
                        ha="right" if m == "o" else "left", va="bottom", fontsize=fs, zorder=7)
    core = [(p, x) for p, x, orb in pk if p < 0 and max(orb, key=orb.get) == "c"]
    if not core:
        return
    qp = max(core, key=lambda q: q[1])
    sats = sorted([q for q in core if q is not qp], key=lambda q: abs(q[0] - qp[0]))
    if lo <= qp[0] <= hi:
        ax.annotate(r"$\varepsilon_{\rm QP}^{c}$", (qp[0], h(qp)), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=fs, zorder=7)
    for n, q in enumerate(sats, 1):
        if lo <= q[0] <= hi:
            ax.annotate(rf"$\varepsilon_{{s{n}}}^{{c}}$", (q[0], h(q)), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=fs, zorder=7)


def bobv_muted(V, tc):
    """Muted poles of the lowest-energy (B_c,B_o,B_v)=(3,3,3) run of test_bo_bv_seeded.py (cons < 1e-9): all eigenvalues of H_qp except the three
    levels of the c, o, v quasiparticle peaks. Returns (poles, E - E_ED)."""
    import exact_cederbaum as ed
    runs = [r for r in np.load(f"cederbaum_data/bobv_seeded_V{V:.4f}_tc{tc}.npy", allow_pickle=True) if r["cons"] < 1e-9]
    r = min(runs, key=lambda x: x["E"])
    sl = [slice(0, 3), slice(3, 6), slice(6, 9)]
    phi = np.linalg.eigh(r["Hqp"])[1]; R = [r["R0"], r["Ro"], r["Rv"]]
    W = np.array([[(R[I] @ phi[sl[I], k]) ** 2 for k in range(9)] for I in range(3)])
    mu = np.ones(9, bool); mu[[np.argmax(W[I]) for I in range(3)]] = False
    return r["eps"][mu], r["E"] - ed.solve(1.0, 10.0, V, tc, ht="cv")["E0"]


def draw_muted(ax, poles, y, color, ms, label=None, tol=0.02):
    """One circle per group of poles closer than tol; the multiplicity is written above the circle when it is larger than 1."""
    P = np.sort(np.asarray(poles)); groups = []
    for p in P:
        if groups and p - groups[-1][-1] < tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    ax.plot([np.mean(g) for g in groups], np.full(len(groups), y), "o", ms=ms, color=color, mec="white", mew=0.8, alpha=0.95, zorder=2, label=label)
    for g in groups:
        if len(g) > 1:
            ax.annotate(str(len(g)), (np.mean(g), y), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, zorder=7)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.load("cederbaum_data/ghost_solutions.npz")
    fig, axs = plt.subplots(len(TCS), len(VS), figsize=(13, 7.4), sharex=True, sharey=True)
    w = np.linspace(-18.0, 14.0, 9000)
    for i, tc in enumerate(TCS):
        for j, V in enumerate(VS):
            ax = axs[i, j]
            ed = ed_poles(V, tc)
            ep = np.concatenate([p for p, _ in ed]); ew = np.concatenate([x for _, x in ed])
            ax.plot(w, lorentz(w, ep, ew), "-", lw=2.2, color="black", label="exact (ED)")
            for Bc, ls, lc, lw in ((3, "--", "red", 2.0), (5, ":", "tab:green", 3.0)):
                eps, W, nocc, comm = gut_poles(V, tc, Bc, Z[f"V{V}_tc{tc}_z{Bc}"])
                ax.plot(w, lorentz(w, eps, W.sum(axis=0)), ls, lw=lw, color=lc, label=rf"gGut ($B_c={Bc}$)")
                qpk = [np.argmax(W[I]) for I in range(3)]                     # the levels of the c, o, v quasiparticle peaks (they carry the weight)
                mu = np.ones(eps.size, bool); mu[qpk] = False                 # muted poles: every other eigenvalue of H_qp, no weight threshold
                draw_muted(ax, eps[mu], MROW[Bc], MC[Bc], MS[Bc], rf"muted poles ($B_c={Bc}$)" if (i, j) == (0, 0) else None)
                if Bc == 3:
                    k = np.argmax(W[0])
                    ax.annotate(r"$\varepsilon_{\rm QP}^{c,{\rm Gut}}$", (eps[k], W[0, k] / (np.pi * ETA)), xytext=(-7, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=11, zorder=7)
                    for I, m, sg in ((1, "o", -1), (2, "v", 1)):
                        k = np.argmax(W[I])
                        ax.annotate(rf"$\varepsilon_{{\rm QP}}^{{{m},{{\rm Gut}}}}$", (eps[k], 0.85 * W[:, k].sum() / (np.pi * ETA)), xytext=(9 * sg, 0),
                                    textcoords="offset points", ha="right" if sg < 0 else "left", va="center", fontsize=11, zorder=7)
                print(f"V={V:.3f} tc={tc} Bc={Bc} comm={comm:.1e}  poles", np.round(eps, 3), " weights/orbital", np.round(W, 5).tolist(), flush=True)
            pv, dE = bobv_muted(V, tc)
            draw_muted(ax, pv, MROW["ov"], MC["ov"], MS["ov"], r"muted poles ($B_{c,o,v}=3$)" if (i, j) == (0, 0) else None)
            print(f"V={V:.3f} tc={tc}: (3,3,3) lowest run E-E_ED={dE:+.2e}, muted poles {np.round(np.sort(pv), 3)}", flush=True)
            label_ed(ax, ed, -18.0, 14.0)
            ax.axvline(0, color="0.7", lw=0.8)
            ax.set_xlim(-18.0, 14.0); ax.set_ylim(-0.68, 2.95)
            ax.set_yticks(np.arange(0, 2.6, 0.5)); ax.tick_params(labelsize=14)
            if i == len(TCS) - 1:
                ax.set_xlabel(r"$\omega/g$", fontsize=17)
            if j == 0:
                ax.set_ylabel(r"$g\,A(\omega)$", fontsize=17)
            ax.text(0.03, 0.93, rf"$\rho/\Omega={np.sqrt(2)*V/2:.1f},\ t_c/g={tc:g}$", transform=ax.transAxes, fontsize=14, va="top")
            if (i, j) == (0, 0):
                ax.legend(fontsize=8, loc="upper right", borderaxespad=0.3, handlelength=1.6)
    plt.tight_layout(w_pad=1.5)
    plt.savefig(PNG, dpi=150); plt.close()
    print("wrote", PNG)


if __name__ == "__main__":
    main()
