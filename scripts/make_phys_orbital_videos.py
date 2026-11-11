"""Movies of the physical (R-transformed) orbitals of the Hubbard dimer (gGA) vs U/t.
The qp orbital phi_n (eigenvector of H_qp, qp basis of both sites) is mapped to the physical space by R: amplitude on physical site s
a_s(n) = R_s . phi_n(site s block) = |R_s| * <center_s|phi_n>  (the ghost components are orthogonal to R_s and drop out).
So sum_n-weight: w_n = 2 (a_0^2 + a_1^2) is the A(w) weight of the movie make_qp_weights_video. Orbitals and phases are the aligned ones of
make_qp_orbital_videos.load_frames (same tracking, same panels). Circle area ~ a_s^2 (full circle = 1), color = sign.
Output: <scan_dir>/phys_orbitals_vs_U.mp4"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import make_qp_orbital_videos as mq
from make_qp_orbital_videos import POS, NEG
OCC_C, VIRT_C = "#0072B2", "#CC79A7"

def phys_amps(f):
    Neff = f["Neff"]; r = [np.linalg.norm(f["R"][s]) for s in range(2)]
    return np.array([r[0] * f["ph"][0, :], r[1] * f["ph"][Neff, :]])      # (2, N)

def make(fr, title, out, fps=12):
    """Physical orbitals = R Phi_qp. R is 2 x 2(1+Ng) (row s = R_s on the block of site s), Phi_qp the 2(1+Ng) x 2(1+Ng) matrix of qp eigenvectors (columns, energy order, tracked
    and phase-aligned as in make_qp_orbital_videos). The product has 2 rows: the 2 physical orbitals phi_phys_s, each given by its coefficients over the qp eigenorbitals n:
    (R Phi)_{s n} = R_s . phi_n(block s). One panel per physical orbital, one circle per qp eigenorbital (left to right in energy order); area ~ coefficient^2, color = sign."""
    N = 2 * fr[0]["Neff"]; nocc = N // 2
    Neff = fr[0]["Neff"]
    xy = mq.skeleton_xy(Neff) if Neff > 1 else np.array([(-1.0, 0.0), (1.0, 0.0)])
    fig = plt.figure(figsize=(max(1.5 * N, 5.0) + 4.2, 4.8)); gs = fig.add_gridspec(2, 2, width_ratios=[max(N, 3.3), 3.3])
    axs = np.array([[fig.add_subplot(gs[s, 0])] for s in range(2)]); axr = [fig.add_subplot(gs[s, 1]) for s in range(2)]
    txt = fig.suptitle("", fontsize=11)
    def update(i):
        f = fr[i]; RPhi = phys_amps(f)                      # (2, N): rows = the two physical orbitals
        for s in range(2):
            ax = axs[s, 0]; ax.clear(); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.7, N - 0.3); ax.set_ylim(-0.8, 0.95)
            ax.plot([0, N - 1], [0, 0], color="0.8", lw=0.8, zorder=1)
            for n in range(N):
                ax.add_patch(Circle((n, 0), 0.45, fill=False, ec="0.75", lw=0.8, zorder=2))
                a = RPhi[s, n]; rad = 0.45 * np.sqrt(min(a * a, 1.0))
                if rad > 1e-3: ax.add_patch(Circle((n, 0), rad, fc=POS if a > 0 else NEG, ec="none", zorder=3))
                ax.text(n, -0.62, rf"$c_{{R_{s + 1},{n + 1}}}$ = {a:+.2f}", ha="center", va="center", fontsize=7.5, color="0.25")
                oc = f["occf"][n] > 0.5
                ax.text(n, 0.80, "occ" if oc else "virt", ha="center", va="center", fontsize=9, fontweight="bold", color=OCC_C if oc else VIRT_C)
                ax.text(n, 0.60, rf"$\epsilon_{{qp}}$ = {f['eps'][n]:+.2f}", ha="center", va="center", fontsize=7.5, color="0.25")
            ax.set_title(f"physical orbital {s + 1}  $\\phi^{{phys}}_{s + 1}$ (row {s + 1} of $R\\Phi^{{qp}}$):  norm$^2$ = {np.sum(RPhi[s] ** 2):.3f}", fontsize=9)
        for s in range(2):    # final physical orbital phi_phys_s = sum_n c_{R_s,n} phi_qp_n: lives on physical site s, the other site is empty
            v = f["ph"] @ RPhi[s]; ax = axr[s]; ax.clear(); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.15, 1.0)
            ax.plot([-1, 1], [0, 0], color="0.6", lw=0.8, zorder=1)
            for k, x in enumerate((-1.0, 1.0)):
                a = v[k * Neff] if k == s else 0.0
                ax.add_patch(Circle((x, 0), 0.8, fill=False, ec="0.75", lw=0.8, zorder=2))
                if abs(a) > 1e-3: ax.add_patch(Circle((x, 0), 0.8 * np.sqrt(min(a * a, 1.0)), fc=POS if a > 0 else NEG, ec="none", zorder=3))
                ax.text(x, -0.93, f"site {k + 1}" + ("" if k == s else " (empty)"), ha="center", va="center", fontsize=8, color="0.3")
            ax.set_title(rf"$\phi^{{phys}}_{s + 1} = \sum_n c_{{R_{s + 1},n}}\,\phi^{{qp}}_n$" + rf"     $\|\phi^{{phys}}_{s + 1}\|^2$ = {np.sum(v ** 2):.4f}  ($\|\phi\|$ = {np.linalg.norm(v):.4f})", fontsize=9)
        txt.set_text(rf"{title}   $U/t$ = {f['U']:.1f}   $Z$ = {f['z']:.3f}" + "\n" + r"$\phi^{phys} = R\,\Phi^{qp}$: coefficients $c_{R_s,n}$ of the physical orbitals over the qp eigenorbitals" + "\n" + r"area $\propto c^2$, color = sign; phases aligned to previous $U/t$")
        return []
    ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    from matplotlib.patches import FancyArrowPatch
    for s_ in range(2):
        bl, br = axs[s_, 0].get_position(), axr[s_].get_position(); y = 0.5 * (bl.y0 + bl.y1); x0, x1 = bl.x1 + 0.005, br.x0 - 0.005
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=2, color="k"))
        fig.text(0.5 * (x0 + x1), y + 0.03, "sum over n\nfinal orbital", ha="center", va="bottom", fontsize=8, transform=fig.transFigure)
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2000)); plt.close(fig)
    print("wrote", out, flush=True)

def ng0_frames(dV):
    """Ng=0 (analytic solution): qp orbitals on the two states, occupied (sqrt D0, sqrt D1), virtual (-sqrt D1, sqrt D0); |R_s| = sqrt(Z) on both sites
    (equal weights 2Z of the two levels force R_0 = R_1). Global signs chosen as in make_ng0_movies.orbitals (those of the Ng=2 HOMO and LUMO)."""
    import make_ng0_movies as mn
    S = mn.states(dV); ref = mn.ng2_homo_lumo(dV); fr = []; last = [1.0, 1.0]
    for U, a, r_ in zip(mn.Us, S, ref):
        D0, D1 = (1 + a["m"]) / 2, (1 - a["m"]) / 2
        o = np.array([np.sqrt(D0), np.sqrt(D1)]); v = np.array([-np.sqrt(D1), np.sqrt(D0)])
        for k, (vec, rr) in enumerate(((o, r_[0]), (v, r_[1]))):
            ov = vec @ rr
            if abs(ov) > 0.3: last[k] = 1.0 if ov > 0 else -1.0
        ph = np.column_stack([last[0] * o, last[1] * v]); Z = a["Z"]
        fr.append(dict(U=U, Neff=1, ph=ph, rho_star=np.outer(ph[:, 0], ph[:, 0]), R=[np.array([np.sqrt(Z)]), np.array([np.sqrt(Z)])], occf=np.array([1.0, 0.0]),
                       eps=np.array([-a["eps"], a["eps"]]), w=np.array([a["w"], a["w"]]), z=Z))
    return fr

if __name__ == "__main__":
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            f = line.split(); br2[round(float(f[0]), 1)] = f[2]
    which = sys.argv[1] if len(sys.argv) > 1 else "dV0"
    dV = 0.0 if which == "dV0" else 2.0
    if which == "dV0": gd2, d2 = (lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}"), "scan_ng2_not_imp"
    else: gd2, d2 = (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}"), "scan_ng2_dV2_confirmed"
    make(ng0_frames(dV), rf"$N_g=0$, $\Delta V={dV:g}$", f"scan_ng0/dV_{dV:.1f}/phys_orbitals_vs_U.mp4")
    make(mq.load_frames(gd2), rf"$N_g=2$, $\Delta V={dV:g}$", f"{d2}/phys_orbitals_vs_U.mp4")
