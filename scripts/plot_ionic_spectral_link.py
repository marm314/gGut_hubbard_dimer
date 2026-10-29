"""Link between the ionic weight and the weight of the inner (nearest to omega=0) hole pole of the spectral function.
Panel A (dV=0): inner weight vs P_ion; exact curve 1+2 sqrt(P(1-P)), ED points, gGA points (P_ion of gGA, weight of its lowest peak), colored by U/t, stopped at U_c.
Panel B (dV=2): inner weight vs U/t: ED, prediction from the ionic/covalent composition of the gGA ground state, and the lowest gGA peak (stopped at U_c).
Writes merged_Ng/ionic_spectral_link_dV0.png and ..._dV2.png (data cached in merged_Ng/ionic_spectral_link_data.npz)."""
import numpy as np
import ionic_covalent as ic
import exact_dimer_deltaV as ed
import gga_dimer as gg
import make_qp_orbital_videos as q
import make_qp_weights_video as mw

t = 1.0
Us = np.round(np.arange(0.1, 10.01, 0.1), 1)
UC = {0: 6.56, 2: 8.34}


def ed_inner(U, dVed):
    """exact weight of the inner hole pole (lowest N=1 orbital) and the exact amplitudes of the ground state"""
    H, c, n0, n1 = ed.build_hamiltonian(t, U, dVed); N = np.real(np.diag(n0 + n1)); dim = H.shape[0]
    sec2 = np.flatnonzero(np.abs(N - 2) < 1e-9); w, v = np.linalg.eigh(H[np.ix_(sec2, sec2)]); gs = np.zeros(dim); gs[sec2] = v[:, 0]
    sec1 = np.flatnonzero(np.abs(N - 1) < 1e-9); w1, v1 = np.linalg.eigh(H[np.ix_(sec1, sec1)])
    cd = [x.conj().T for x in c]
    M = sum(x.conj().T @ x for x in c); vac = np.linalg.eigh(M)[1][:, 0]
    basis = ((cd[0] @ cd[3] - cd[1] @ cd[2]) @ vac / np.sqrt(2), cd[0] @ cd[1] @ vac, cd[2] @ cd[3] @ vac)    # cov, ion on site 0, ion on site 1
    amp = np.array([b @ gs for b in basis])

    def inner(state):
        wt = np.zeros(len(sec1))
        for op in c: wt += (v1.T @ (op @ state)[sec1]) ** 2
        lo = np.isclose(w1, w1.min(), atol=1e-9)
        return wt[lo].sum()
    return inner(gs), inner, basis, amp


def gga_site_docc(d, U, dV):
    R, lam = q.load_R_lambda(d); Neff = len(R[0])
    ops = [gg.build_impurity_ops(Neff, U, U / 2 + dV), gg.build_impurity_ops(Neff, U, U / 2 - dV)]
    Hq, rho = gg.qp_state(R, lam, t); bl = [rho[:Neff, :Neff], rho[Neff:, Neff:]]
    V = [gg.hybridization_V(rho, R, t, 1e-9, i) for i in range(2)]
    lc = [gg.embedding_lambda_c(bl[i], lam[i], V[i], R[i], 1e-9) for i in range(2)]
    imp = [gg.impurity_solve(V[i], lc[i], ops[i]) for i in range(2)]
    Lb = np.zeros_like(Hq); Lb[:Neff, :Neff] = lam[0]; Lb[Neff:, Neff:] = lam[1]
    return [imp[0][3], imp[1][3]], 2.0 * np.trace(rho @ (Hq + Lb)) + imp[0][2] + imp[1][2]


def gga_inner_peak(f):
    neg = [(e, w) for e, w in zip(f["eps"], f["w"]) if e < -1e-9 and w >= 1e-3]
    return max(neg)[1] if neg else np.nan          # the weighted negative-energy level nearest to epsilon = 0


def compute():
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            f = line.split(); br2[round(float(f[0]), 1)] = f[2]
    out = {"U": Us}
    for dV in (0, 2):
        dirf = (lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}") if dV == 0 else (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}")
        fr = {f["U"]: f for f in q.load_frames(dirf)}
        edw, pion_ed, pred, peak, pion_g = [], [], [], [], []
        for U in Us:
            w_ed, inner, basis, amp = ed_inner(U, 2.0 * dV)
            doc, _ = gga_site_docc(dirf(U), U, float(dV))
            dED = [doc[1], doc[0]] if dV > 0 else doc          # ED site 0 is the depleted one for dV > 0
            sg = np.sign(amp); sg[sg == 0] = 1
            cs = np.array([np.sqrt(max(1 - dED[0] - dED[1], 0)) * sg[0], np.sqrt(dED[0]) * sg[1], np.sqrt(dED[1]) * sg[2]])
            st = sum(cs[i] * basis[i] for i in range(3)); st /= np.linalg.norm(st)
            edw.append(w_ed); pred.append(inner(st)); pion_g.append(doc[0] + doc[1]); peak.append(gga_inner_peak(fr[U]))
            pion_ed.append(ic.ed_weights(U, 2.0 * dV)[0])
            print(f"dV={dV} U/t={U:.1f}  ED inner={w_ed:.4f}  predicted from gGA composition={pred[-1]:.4f}  gGA peak={peak[-1]:.4f}", flush=True)
        out.update({f"ed_inner_dV{dV}": np.array(edw), f"pion_ed_dV{dV}": np.array(pion_ed), f"pred_dV{dV}": np.array(pred),
                    f"peak_dV{dV}": np.array(peak), f"pion_gga_dV{dV}": np.array(pion_g)})
    np.savez("merged_Ng/ionic_spectral_link_data.npz", **out)


def plot():
    """One figure per Delta V, same layout: left, inner weight vs U/t (ED, exact relation with the gGA composition, lowest gGA peak);
    right, inner weight vs P_ion (ED, and the exact curve at dV=0; gGA points colored by U/t). gGA stops at U_c."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    d = np.load("merged_Ng/ionic_spectral_link_data.npz"); U = d["U"]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    cm = LinearSegmentedColormap.from_list("blues_t", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
    for dV in (0, 2):
        fig, (bx, ax) = plt.subplots(1, 2, figsize=(12.2, 4.8))
        m = U < UC[dV]
        # left: vs U/t
        bx.plot(U, d[f"ed_inner_dV{dV}"], color="0.1", lw=2.2, label="ED (exact)")
        bx.plot(U, d[f"pred_dV{dV}"], color="0.45", lw=1.8, ls=(0, (4, 3)), label="exact relation with the gGA ionic/covalent composition")
        bx.plot(U[m], d[f"peak_dV{dV}"][m], color="#0072B2", lw=2.0, label=r"gGA, lowest peak ($U<U_c$)")
        bx.axvline(UC[dV], color="0.55", lw=1.0, ls=":"); bx.text(UC[dV] + 0.1, 1.9, rf"$U_c/t={UC[dV]}$", color="0.35", fontsize=11, va="top")
        bx.set_xlabel(r"$U/t$"); bx.set_ylabel("weight of the inner hole pole"); bx.set_xlim(0, 10); bx.set_ylim(0, 2.1)
        bx.legend(frameon=False, loc="lower left", fontsize=10)
        # right: vs P_ion
        if dV == 0:
            pp = np.linspace(1e-4, 0.5, 400); ax.plot(pp, 1 + 2 * np.sqrt(pp * (1 - pp)), color="0.1", lw=2.0, label=r"exact: $1+2\sqrt{P_{\rm ion}(1-P_{\rm ion})}$")
        ax.scatter(d[f"pion_ed_dV{dV}"], d[f"ed_inner_dV{dV}"], s=14, color="0.1", zorder=3, label="ED (one point per $U/t$)")
        sc = ax.scatter(d[f"pion_gga_dV{dV}"][m], d[f"peak_dV{dV}"][m], c=U[m], cmap=cm, vmin=0, vmax=UC[dV], s=26, edgecolor="#0072B2", lw=0.5, zorder=4,
                        label=r"gGA, lowest peak ($U<U_c$)")
        for u in (2, 4, 6):
            i = list(U).index(float(u))
            ax.annotate(rf"$U/t={u}$", (d[f"pion_ed_dV{dV}"][i], d[f"ed_inner_dV{dV}"][i]), textcoords="offset points", xytext=(-8, 10), fontsize=9, color="0.25", ha="right")
            ax.annotate(rf"$U/t={u}$", (d[f"pion_gga_dV{dV}"][i], d[f"peak_dV{dV}"][i]), textcoords="offset points", xytext=(8, -4), fontsize=9, color="#0072B2")
        j = np.flatnonzero(m)[-1]
        ax.annotate(rf"gGA stops at $U_c/t={UC[dV]}$", (d[f"pion_gga_dV{dV}"][j], d[f"peak_dV{dV}"][j]), textcoords="offset points", xytext=(14, -2), fontsize=10, color="#0072B2",
                    arrowprops=dict(arrowstyle="-", color="#0072B2", lw=0.8))
        ax.set_xlabel(r"ionic weight $P_{\rm ion}$ (right: small $U/t$, left: large $U/t$)"); ax.set_ylabel("weight of the inner hole pole")
        ax.set_xlim(0, 1.0 if dV == 2 else 0.52); ax.set_ylim(0, 2.1); ax.legend(frameon=False, loc="lower right", fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="$U/t$ (gGA)")
        fig.suptitle(rf"$\Delta V={dV}$", fontsize=14)
        plt.tight_layout(); out = f"merged_Ng/ionic_spectral_link_dV{dV}.png"; plt.savefig(out, dpi=150); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compute": compute()
    else: plot()
