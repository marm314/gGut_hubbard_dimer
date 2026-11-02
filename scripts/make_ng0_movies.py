"""Ng=0 movies from the ANALYTIC solution (scripts/ng0_analytic.py), for dV=0 and dV=2 (U/t=0.1..10, step 0.1), written to scan_ng0/dV_<dV>/:
  qp_weights_vs_U.mp4  the two quasiparticle levels +-E_q (E_q = tZ/sqrt(1-m^2)) with weight 2Z each, against the ED poles;
  qp_orbitals_vs_U.mp4 the occupied (bonding) and virtual (antibonding) quasiparticle orbitals on the two states, circle area |amplitude|^2, colour = sign (global signs chosen to match the Ng=2 HOMO and LUMO);
  qp_density_vs_U.mp4  quasiparticle density on the two states, spectral gap, bond order, Z and double occupancy (dV=0) / state densities (dV=2), with U_c.
(A_omega_vs_U.mp4 comes from make_ng0_videos.py.) usage (repo root): python3 scripts/make_ng0_movies.py [weights|orbitals|density ...]"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ng0_analytic import ng0_state
import make_qp_weights_video as mw
t = 1.0; Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]
BLUE, RED, POS, NEG = "#0072B2", "crimson", "#d95f02", "#1b9e77"
def uc(dV): return 4 * t + 2 * np.sqrt(4 * t * t + dV * dV)
def states(dV): return [ng0_state(U, dV) for U in Us]
def save(fig, upd, out): animation.FuncAnimation(fig, upd, frames=len(Us), blit=False).save(out, writer=animation.FFMpegWriter(fps=12, bitrate=1800)); plt.close(fig); print("wrote", out, flush=True)
def weights(dV, S, out):
    ed = mw.ed_poles0 if dV == 0 else mw.ed_poles2
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); ax.set_xlim(-8, 8); ax.set_ylim(0, 2.3); ax.set_xlabel("quasiparticle energy / t"); ax.set_ylabel("level weight")
    ax.set_title(rf"$N_g=0$, $\Delta V={dV:g}$: qp levels $\pm tZ/\sqrt{{1-m^2}}$, weight $2Z$"); txt = ax.text(0.03, 0.9, "", transform=ax.transAxes, fontsize=13); arts = []
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", lw=3, label="gGA $N_g=0$"), plt.Line2D([], [], color=RED, marker="o", mfc="none", ls="", ms=8, label="ED pole")], loc="upper right", frameon=False)
    def upd(i):
        nonlocal arts
        for a in arts: a.remove()
        a = S[i]; pe, we = ed(Us[i]); e = np.array([-a["eps"], a["eps"]]); w = np.array([a["w"], a["w"]]); col = BLUE if a["Z"] > 0 else "lightgray"
        arts = [ax.vlines(e, 0, w, colors=col, lw=3), ax.scatter(e, w, s=34, color=col, zorder=3), ax.scatter(pe, we, s=70, facecolors="none", edgecolors=RED, lw=1.4, zorder=4)]
        txt.set_text(f"U/t = {Us[i]:.1f}   Z = {a['Z']:.3f}" + ("   (Z = 0: no weight)" if a["Z"] == 0 else "")); return arts
    plt.tight_layout(); save(fig, upd, out)
def ng2_homo_lumo(dV):
    """(homo, lumo) amplitudes on the two states (centre components) of the tracked, phase-aligned Ng=2 orbitals, per U/t: the sign reference for the Ng=0 orbitals"""
    import make_qp_orbital_videos as mo
    if dV == 0:
        br2 = {}
        for line in open("scan_ng2_not_imp/lowest_energy.txt"):
            if not line.startswith("#") and line.strip(): f = line.split(); br2[round(float(f[0]), 1)] = f[2]
        fr = mo.load_frames(lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}")
    else: fr = mo.load_frames(lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}")
    out = []
    for f in fr:
        Neff = f["Neff"]; occ = f["occf"] > 0.5; ih = int(np.argmax(np.where(occ, f["eps"], -np.inf))); il = int(np.argmin(np.where(~occ, f["eps"], np.inf)))
        out.append((f["ph"][[0, Neff], ih], f["ph"][[0, Neff], il]))
    return out
def orbitals(dV, S, out):
    ref = ng2_homo_lumo(dV)
    # global signs as the Ng=2 HOMO and LUMO, but only where that orbital has a clear overlap (>0.3) with the Ng=0 one on the two states: where the Ng=2 HOMO/LUMO is a ghost level
    # (small U/t at dV=2, above U_c) its sign relative to Ng=0 is meaningless, so the sign of the previous frame is kept (no colour flicker)
    signs = []
    for a_, r_ in zip(S, ref):
        D0_, D1_ = (1 + a_["m"]) / 2, (1 - a_["m"]) / 2
        ov_ = (np.array([np.sqrt(D0_), np.sqrt(D1_)]) @ r_[0], np.array([-np.sqrt(D1_), np.sqrt(D0_)]) @ r_[1])
        signs.append([(1.0 if ov_[k] > 0 else -1.0) if abs(ov_[k]) > 0.3 else None for k in range(2)])
    for k in range(2):                                              # hold the last clear sign; the frames before the first clear one take it too
        first = next((sg[k] for sg in signs if sg[k] is not None), 1.0); last = first
        for sg in signs:
            if sg[k] is None: sg[k] = last
            last = sg[k]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.9)); ttl = fig.suptitle("", fontsize=13)
    def upd(i):
        a = S[i]; D0, D1 = (1 + a["m"]) / 2, (1 - a["m"]) / 2
        o = np.array([np.sqrt(D0), np.sqrt(D1)]); v = np.array([-np.sqrt(D1), np.sqrt(D0)]); sg_ = signs[i]
        o = sg_[0] * o; v = sg_[1] * v
        for ax, (u0, u1), name, e in ((axs[0], tuple(o), "occupied (bonding)", -a["eps"]), (axs[1], tuple(v), "virtual (antibonding)", a["eps"])):
            ax.clear(); ax.axis("off"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.6, 0.6); ax.set_aspect("equal")
            ax.plot([-0.1, 0.1], [0, 0], color="0.5", lw=2, zorder=0)
            for x, u in ((-0.45, u0), (0.45, u1)):
                ax.add_patch(plt.Circle((x, 0), 0.32, fill=False, ec="0.75", lw=1)); ax.add_patch(plt.Circle((x, 0), 0.32 * abs(u), color=POS if u >= 0 else NEG))
            ax.set_title(f"{name}\n$\\epsilon={e:+.2f}$,  $A(\\omega)$ weight $={a['w']:.2f}$", fontsize=11)
        ttl.set_text(rf"$N_g=0$, $\Delta V={dV:g}$,  $U/t={Us[i]:.1f}$,  $Z={a['Z']:.3f}$"); return []
    fig.text(0.5, 0.02, r"circles: the two states; area $\propto|\mathrm{amplitude}|^2$, colour = sign", ha="center", fontsize=10, color="0.35")
    plt.tight_layout(rect=(0, 0.06, 1, 0.9)); save(fig, upd, out)
def density(dV, S, out):
    Uc = uc(dV); m = np.array([a["m"] for a in S]); Z = np.array([a["Z"] for a in S]); eps = np.array([a["eps"] for a in S])
    fig = plt.figure(figsize=(11, 5.4)); gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1]); ax0 = fig.add_subplot(gs[:, 0]); axa, axb, axc = (fig.add_subplot(gs[k, 1]) for k in range(3))
    axa.plot(Us, 2 * eps, label="spectral gap = qp gap $2E_q$"); axa.set_ylabel("gap"); axa.legend(fontsize=8, frameon=False)
    axb.plot(Us, np.sqrt(1 - m ** 2) / 2, color="C2", label=r"bond order $\rho_{01}$"); axb.plot(Us, Z, color="C3", label="Z"); axb.legend(fontsize=8, frameon=False)
    if dV == 0: axc.plot(Us, [a["docc"] for a in S], color="C4", label="double occupancy")
    else: axc.plot(Us, (1 + m) / 2, color="C0", label="state density (site 0)"); axc.plot(Us, (1 - m) / 2, color="C1", ls="--", label="state density (site 1)")
    axc.legend(fontsize=8, frameon=False); axc.set_xlabel("U/t")
    for ax in (axa, axb, axc): ax.axvline(Uc, color="crimson", ls="--", lw=1.2); ax.set_xlim(0, 10)
    axa.text(Uc - 0.15, axa.get_ylim()[1] * 0.5, rf"$U_c/t={Uc:.2f}$", color="crimson", fontsize=9, ha="right", va="top")
    marks = [ax.axvline(Us[0], color="k", lw=1) for ax in (axa, axb, axc)]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1)); fig.colorbar(sm, ax=ax0, fraction=0.04, label="qp density per spin")
    txt = fig.suptitle("")
    def upd(i):
        a = S[i]; ax0.clear(); ax0.axis("off"); ax0.set_xlim(-1, 1); ax0.set_ylim(-0.7, 0.7); ax0.set_aspect("equal")
        for x, d in ((-0.45, (1 + a["m"]) / 2), (0.45, (1 - a["m"]) / 2)): ax0.add_patch(plt.Circle((x, 0), 0.32, color=plt.cm.viridis(d))); ax0.text(x, 0, f"{d:.2f}", ha="center", va="center", color="w" if d < 0.6 else "k", fontsize=12)
        ax0.plot([-0.13, 0.13], [0, 0], color="0.4", lw=3, zorder=0)
        for mk in marks: mk.set_xdata([Us[i], Us[i]])
        txt.set_text(rf"$N_g=0$, $\Delta V={dV:g}$,  $U/t={Us[i]:.1f}$,  $Z={a['Z']:.3f}$"); return []
    plt.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, upd, out)
if __name__ == "__main__":
    what = sys.argv[1:] or ["weights", "orbitals", "density"]
    for dV in (0.0, 2.0):
        S = states(dV); d = f"scan_ng0/dV_{dV:.1f}"
        if "weights" in what: weights(dV, S, f"{d}/qp_weights_vs_U.mp4")
        if "orbitals" in what: orbitals(dV, S, f"{d}/qp_orbitals_vs_U.mp4")
        if "density" in what: density(dV, S, f"{d}/qp_density_vs_U.mp4")
