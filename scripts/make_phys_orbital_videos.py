"""Movies of the physical (R-transformed) orbitals of the Hubbard dimer (gGA) vs U/t.
The eigenorbital phi^qp_n = sum_i chi_i (C_qp)_{in} of H_qp (chi_i = site+ghost basis, here the star basis: for each site the first chi is the "state", the others the ghosts)
is mapped to the physical space by R: C_{R_s,n} = (R C_qp)_{sn} = sum_i R_{si} (C_qp)_{in} = |R_s| (C_qp)_{state(s),n}, since in the star basis R_{si} = |R_s| delta_{i,state(s)}
(the ghost components are orthogonal to R_s and drop out). Physical orbital: phi^phys_s = sum_n C_{R_s,n} phi^qp_n = sum_i R_{si} chi_i.
So sum_n-weight: w_n = 2 (C_{R_1,n}^2 + C_{R_2,n}^2) is the A(w) weight of the movie make_qp_weights_video. Orbitals and phases are the aligned ones of
make_qp_orbital_videos.load_frames (same tracking, same panels). Circle area ~ C_{R_s,n}^2 (full circle = 1), color = sign.
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

def phys_amps(f, chip=False):
    if chip:    # chi' basis: R'_s = (|R_s|,0,..), so C_{R_s,n} = |R_s| (C'_qp)_{1n} (the state row of the tracked star eigenvectors)
        Neff = f["Neff"]; return np.array([np.linalg.norm(f["R"][s]) * f["ph"][s * Neff, :] for s in range(2)])
    Neff = f["Neff"]; X = f.get("phx", f["ph"])                           # phi^qp_n = sum_i chi_i (C_qp)_{in}, chi = solver basis
    return np.array([f.get("Rchi", f["R"])[s] @ X[s * Neff:(s + 1) * Neff, :] for s in range(2)])      # (2, N): C_{R_s,n} = sum_i R_{si} (C_qp)_{in}

def make(fr, title, out, fps=12, chip=False):
    """Physical orbitals = R C_qp. R is 2 x 2(1+Ng) (row s = R_{si} on the block of site s), C_qp the 2(1+Ng) x 2(1+Ng) matrix of qp eigenvectors (columns, energy order, tracked
    and phase-aligned as in make_qp_orbital_videos). The product has 2 rows: the 2 physical orbitals phi_phys_s, each given by its coefficients over the qp eigenorbitals n:
    (R Phi)_{s n} = sum_i R_{si} (C_qp)_{in}. One panel per physical orbital, one circle per qp eigenorbital (left to right in energy order); area ~ coefficient^2, color = sign."""
    N = 2 * fr[0]["Neff"]; nocc = N // 2
    Neff = fr[0]["Neff"]
    xy = mq.skeleton_xy(Neff) if Neff > 1 else np.array([(-1.0, 0.0), (1.0, 0.0)])
    fig = plt.figure(figsize=(max(1.7 * N, 5.0) + 6.0, 7.4)); gs = fig.add_gridspec(2, 2, width_ratios=[max(N, 3.3), 4.6], hspace=0.5)
    axs = np.array([[fig.add_subplot(gs[s, 0])] for s in range(2)]); axr = [fig.add_subplot(gs[s, 1]) for s in range(2)]
    txt = fig.suptitle("", fontsize=14)
    def update(i):
        f = fr[i]; RPhi = phys_amps(f, chip)                      # (2, N): rows = the two physical orbitals
        for s in range(2):
            ax = axs[s, 0]; ax.clear(); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.7, N - 0.3); ax.set_ylim(-0.8, 0.95)
            ax.plot([0, N - 1], [0, 0], color="0.8", lw=0.8, zorder=1)
            for n in range(N):
                ax.add_patch(Circle((n, 0), 0.45, fill=False, ec="0.75", lw=0.8, zorder=2))
                a = RPhi[s, n]; rad = 0.45 * np.sqrt(min(a * a, 1.0))
                if rad > 1e-3: ax.add_patch(Circle((n, 0), rad, fc=POS if a > 0 else NEG, ec="none", zorder=3))
                ax.text(n, -0.62, rf"$C_{{R_{s + 1},{n + 1}}}$ = {a:+.2f}", ha="center", va="center", fontsize=10, color="0.2")
                oc = f["occf"][n] > 0.5
                ax.text(n, 0.80, "occ" if oc else "virt", ha="center", va="center", fontsize=12, fontweight="bold", color=OCC_C if oc else VIRT_C)
                ax.text(n, 0.60, rf"$\epsilon_{{qp}}$ = {f['eps'][n]:+.2f}", ha="center", va="center", fontsize=10, color="0.2")
            ax.set_title(rf"$\phi^{{phys}}_{s + 1}$ in the $\phi^{{qp}}_n$ basis:  $\sum_n C_{{R_{s + 1},n}}^2$ = {np.sum(RPhi[s] ** 2):.3f}", fontsize=12)
        for s in range(2):    # final physical orbital: only the block of site s is non-empty
            ax = axr[s]; ax.clear(); amp = np.zeros(2 * Neff)
            if chip:      # chi' basis: R'_s = (|R_s|, 0, ..): the whole physical orbital is the state chi'_1 of site s
                amp[s * Neff] = np.sum(np.asarray(f["R"][s]) ** 2)
                mq.draw_sk(ax, xy, Neff, amp, scale=(0.3 if Neff > 1 else 0.38), labels=[rf"$\chi'_{{{k_ + 1}}}$" for k_ in range(2 * Neff)])
                ax.set_title(rf"$\phi^{{phys}}_{s + 1} = \sum_j R'_{{{s + 1}j}}\,\chi'_j = |R_{s + 1}|\,\chi'_{{{s * Neff + 1}}}$" + "\n" + rf"$\|\phi^{{phys}}_{s + 1}\|^2 = |R_{s + 1}|^2$ = {np.sum(np.asarray(f['R'][s]) ** 2):.4f}", fontsize=12)
                continue
            Rs = np.asarray(f.get("Rchi", f["R"])[s]); amp[s * Neff:(s + 1) * Neff] = Rs ** 2 * np.sign(Rs)
            mq.draw_sk(ax, xy, Neff, amp, chi=True, scale=(0.3 if Neff > 1 else 0.38))
            ax.set_title(rf"$\phi^{{phys}}_{s + 1} = \sum_i R_{{{s + 1}i}}\,\chi_i$" + "\n" + rf"$\|\phi^{{phys}}_{s + 1}\|^2 = \sum_i R_{{{s + 1}i}}^2$ = {np.sum(f['R'][s] ** 2):.4f}", fontsize=12)
        if chip:
            txt.set_text(rf"{title}   $U/t$ = {f['U']:.1f}   $Z$ = {f['z']:.3f}" + "\n" + r"$\chi'_j$: star basis ($\chi'_1,\chi'_{%d}$ = states, direction of $R_s$; others = ghosts);   $\phi^{qp}_n=\sum_j\chi'_j\,(C'_{qp})_{jn}$: eigenorbitals of $H_{qp}$ (occ/virt, $\epsilon_{qp}$)" % (Neff + 1) + "\n" + r"$\phi^{phys}_s=\sum_n C_{R_s,n}\,\phi^{qp}_n=|R_s|\chi'_{1}$,   $C_{R_s,n}=|R_s|\,(C'_{qp})_{1n}$;   area $\propto C_{R_s,n}^2$, color = sign")
        else:
            txt.set_text(rf"{title}   $U/t$ = {f['U']:.1f}   $Z$ = {f['z']:.3f}" + "\n" + r"$\chi_i$ ($i=1..2(1+N_g)$): solver basis, only the order of the $\chi_i$ within each site is changed;   $\phi^{qp}_n=\sum_i\chi_i\,(C_{qp})_{in}$: eigenorbitals of $H_{qp}$ (occ/virt, $\epsilon_{qp}$)" + "\n" + r"$\phi^{phys}_s=\sum_n C_{R_s,n}\,\phi^{qp}_n=\sum_i R_{si}\chi_i$,   $C_{R_s,n}=(R\,C_{qp})_{sn}$;   area $\propto C_{R_s,n}^2$, color = sign")
        return []
    ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.72, bottom=0.04, hspace=0.75, wspace=0.12)
    from matplotlib.patches import FancyArrowPatch
    for s_ in range(2):
        bl, br = axs[s_, 0].get_position(), axr[s_].get_position(); y = 0.5 * (bl.y0 + bl.y1); x0, x1 = bl.x1 + 0.005, br.x0 - 0.005
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=2, color="k"))
        fig.text(0.5 * (x0 + x1), y + 0.03, "sum over n\nfinal orbital", ha="center", va="bottom", fontsize=11, transform=fig.transFigure)
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
    if len(sys.argv) > 2 and sys.argv[2] == "chip":      # chi' (star) basis set: Ng=2 only (at Ng=0 chi' = chi)
        make(mq.load_frames(gd2), rf"$N_g=2$, $\Delta V={dV:g}$", f"{d2}/phys_orbitals_vs_U_chip.mp4", chip=True)
        sys.exit(0)
    make(ng0_frames(dV), rf"$N_g=0$, $\Delta V={dV:g}$", f"scan_ng0/dV_{dV:.1f}/phys_orbitals_vs_U.mp4")
    make(mq.perm_only_frames(mq.load_frames(gd2)), rf"$N_g=2$, $\Delta V={dV:g}$", f"{d2}/phys_orbitals_vs_U.mp4")
