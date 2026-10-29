"""Videos of the quasiparticle orbitals of the Hubbard dimer (gGA, Delta V = 0) as U/t grows, drawn on the star skeleton of
$HOME/a.jpeg (a physical-like 'center' orbital per site with Ng satellite (ghost) orbitals attached to it; the two centers are linked).

Star basis of a site s: center c_s = R_s/|R_s| (the only qp orbital hopping to the other site, with -t |R_0||R_1|), and the satellites are the
eigenvectors of the on-site block H_s = -lambda_s projected on the orthogonal complement of c_s (sorted by energy), coupled to the center by v_a.
Half filling, as in gga_dimer.qp_state: the lowest N/2 of the N = 2(1+Ng) eigenvectors returned by eigh are filled (per spin), no fractional
occupation; the density is the code's own 1-RDM rho in the star basis.

orbitals video : one panel per eigenorbital of H_qp (top row occupied, bottom row empty): circle area ~ |amplitude|^2, color = sign.
density video  : qp density rho_aa = sum_occ |phi(a)|^2 on the skeleton, bond order between centers, and gaps / Z / double occupancy vs U/t.
Writes <scan_dir>/qp_orbitals_vs_U.mp4 and <scan_dir>/qp_density_vs_U.mp4 (ffmpeg)."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

import glob
import os
from plot_eqp_3d import t


def load_R_lambda(d):
    """as plot_eqp_3d.load_R_lambda, but with the first file in sorted order when a directory holds several equivalent solutions"""
    R = np.loadtxt(sorted(glob.glob(os.path.join(d, "*_R.dat")))[0])
    lam = np.loadtxt(sorted(glob.glob(os.path.join(d, "*_lambda.dat")))[0])
    Neff = R.shape[1]
    return [R[0], R[1]], [lam[:Neff], lam[Neff:]]
import make_qp_weights_video as mw
import gga_dimer as gga

Us = mw.Us
POS, NEG = "#d95f02", "#1b9e77"


def star(R, lam):
    """Transformation Tmat (dim x dim) from the original qp basis to the star basis [center0, sat0..., center1, sat1...] and the star
    couplings (center energy, satellite energies, center-satellite couplings, center-center hopping)."""
    Neff = len(R[0]); dim = 2 * Neff
    T = np.zeros((dim, dim)); info = []
    for s in range(2):
        H = -lam[s]; r = R[s]; c = r / np.linalg.norm(r)
        # orthonormal complement
        q, _ = np.linalg.qr(np.column_stack([c, np.eye(Neff)])); Q = q[:, 1:Neff]
        e, Y = np.linalg.eigh(Q.T @ H @ Q)
        S = Q @ Y
        v = S.T @ H @ c
        S = S * np.where(v < 0, -1.0, 1.0)[None, :]              # canonical sign: center-satellite coupling >= 0
        cols = np.column_stack([c, S])
        T[s * Neff:(s + 1) * Neff, s * Neff:(s + 1) * Neff] = cols
        info.append((float(c @ H @ c), e, S.T @ H @ c))
    tc = -t * np.linalg.norm(R[0]) * np.linalg.norm(R[1])
    return T, info, tc


def skeleton_xy(Neff):
    """positions of the star orbitals: [center0, sat0..., center1, sat1...]"""
    Ng = Neff - 1; xy = []
    for s, sx in enumerate((-1.0, 1.0)):
        xy.append((sx, 0.0))
        ang = np.radians(np.linspace(70, 290, Ng)) if Ng > 1 else np.radians([180.0])
        for a in ang:
            dx, dy = np.cos(a), np.sin(a)
            xy.append((sx + (dx if s == 0 else -dx) * 0.62, dy * 0.62 * (1.5 if Ng > 2 else 1.0)))
    return np.array(xy)


def draw_sk(ax, xy, Neff, amp, color_by_sign=True, vals=None, scale=0.26, bonds=None, cmap=None):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(xy[:, 0].min() - 0.45, xy[:, 0].max() + 0.45); ax.set_ylim(xy[:, 1].min() - 0.45, xy[:, 1].max() + 0.45)
    cen = (0, Neff)
    for s in range(2):
        for a in range(1, Neff):
            i, j = s * Neff, s * Neff + a
            lw = 0.6 if bonds is None else min(bonds[(i, j)], 3.5)
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.55", lw=lw, zorder=1)
    lw = 0.6 if bonds is None else min(bonds[(0, Neff)], 3.5)
    ax.plot([xy[0, 0], xy[Neff, 0]], [xy[0, 1], xy[Neff, 1]], color="0.55", lw=lw, zorder=1)
    for k in range(2 * Neff):
        big = k % Neff == 0
        rad = scale * (1.35 if big else 1.0)
        ax.add_patch(Circle(xy[k], rad, fill=False, ec="0.75", lw=0.8, zorder=2))
        if color_by_sign:
            r = rad * np.sqrt(min(abs(amp[k]), 1.0))
            if r > 1e-3: ax.add_patch(Circle(xy[k], r, fc=POS if amp[k] > 0 else NEG, ec="none", zorder=3))
        else:
            ax.add_patch(Circle(xy[k], rad * 0.98, fc=cmap(vals[k]), ec="none", zorder=3))


def load_frames(get_dir):
    fr = []
    for U in Us:
        d = get_dir(U)
        R, lam = load_R_lambda(d)
        Neff = len(R[0]); dim = 2 * Neff
        H = np.zeros((dim, dim)); H[:Neff, :Neff] = -lam[0]; H[Neff:, Neff:] = -lam[1]
        H[:Neff, Neff:] = -t * np.outer(R[0], R[1]); H[Neff:, :Neff] = H[:Neff, Neff:].T
        eps, phi = np.linalg.eigh(H)
        Rm = np.zeros((dim, 2)); Rm[:Neff, 0] = R[0]; Rm[Neff:, 1] = R[1]
        w = 2.0 * ((Rm.T @ phi) ** 2).sum(axis=0)
        T, info, tc = star(R, lam)
        ph = T.T @ phi                                  # eigenvectors in the star basis (columns)
        nocc = dim // 2; occf = np.zeros(dim); occf[:nocc] = 1.0   # as gga_dimer.qp_state: the lowest Neff eigenvectors returned by eigh, filled
        _, rho0 = gga.qp_state(R, lam, t)                                      # the code's own 1-RDM (original basis), no fractional filling
        rho_star = T.T @ rho0 @ T
        for n in range(dim):                            # gauge: largest component positive, center0 preferred
            ref = ph[0, n] if abs(ph[0, n]) > 0.05 else ph[np.argmax(np.abs(ph[:, n])), n]
            if ref < 0: ph[:, n] *= -1
        line = open(d + "/output.out").read()
        z = float(re.search(r"Z0=([\d.eE+-]+)", line).group(1)); md = re.search(r"docc=([\d.eE+-]+)", line); docc = float(md.group(1)) if md else np.nan
        fr.append(dict(U=U, rho_star=rho_star, occf=occf, eps=eps, ph=ph, w=w, info=info, tc=tc, z=z, docc=docc, Neff=Neff, R=R))
    align_frames(fr)
    return fr


def align_frames(fr):
    """Make the orbitals continuous in U/t. (i) Satellites: the star satellites of a site are re-identified between consecutive U/t
    (Hungarian on |d e| + |d v|) so a satellite keeps its position on the skeleton; the eigenvector components are permuted accordingly.
    (ii) Signs: maximum-overlap partner in the previous frame, positive overlap (panels stay in energy order)."""
    from scipy.optimize import linear_sum_assignment
    Neff = fr[0]["Neff"]; Ng = Neff - 1
    for k in range(1, len(fr)):
        p, c_ = fr[k - 1], fr[k]
        perm = np.arange(2 * Neff)
        for s_ in range(2):
            ep, vp = p["info"][s_][1], np.abs(p["info"][s_][2]); ec, vc = c_["info"][s_][1], np.abs(c_["info"][s_][2])
            cost = np.abs(ep[:, None] - ec[None, :]) + np.abs(vp[:, None] - vc[None, :])
            r, cc = linear_sum_assignment(cost)                 # previous satellite r <-> current satellite cc
            for a, b in zip(r, cc): perm[s_ * Neff + 1 + a] = s_ * Neff + 1 + b
        # new position i of a component = old index perm[i]
        c_["ph"] = c_["ph"][perm, :]
        c_["info"] = [(c_["info"][s_][0], c_["info"][s_][1][perm[s_ * Neff + 1:(s_ + 1) * Neff] - s_ * Neff - 1],
                       c_["info"][s_][2][perm[s_ * Neff + 1:(s_ + 1) * Neff] - s_ * Neff - 1]) for s_ in range(2)]
        # phases: orbital n (energy order) takes the sign that makes its overlap with its maximum-overlap partner of the previous frame
        # positive (MOM for the signs only; the panels stay in energy order)
        ov = p["ph"].T @ c_["ph"]                               # <prev_m | cur_n>
        for n in range(ov.shape[1]):
            m = np.argmax(np.abs(ov[:, n]))
            if ov[m, n] < 0: c_["ph"][:, n] *= -1


def bonds_from(f):
    Neff = f["Neff"]; b = {}
    scale = 4.0
    for s in range(2):
        v = f["info"][s][2]
        for a in range(1, Neff): b[(s * Neff, s * Neff + a)] = max(0.3, scale * abs(v[a - 1]))
    b[(0, Neff)] = max(0.3, scale * abs(f["tc"]))
    return b


def make_orbitals(fr, title, out, fps=12):
    Neff = fr[0]["Neff"]; N = 2 * Neff; xy = skeleton_xy(Neff); nocc = N // 2
    fig, axs = plt.subplots(2, nocc, figsize=(3.4 * nocc, 6.2)); axs = np.atleast_2d(axs)
    txt = fig.suptitle("", fontsize=14)

    def update(i):
        f = fr[i]
        for n in range(N):
            ax = axs[n // nocc, n % nocc]; ax.clear()
            amp = f["ph"][:, n] ** 2 * np.sign(f["ph"][:, n])
            draw_sk(ax, xy, Neff, amp, bonds=bonds_from(f))
            fo = f["occf"][n]
            lab = "occ" if fo > 0.5 else "empty"
            ax.set_title(f"{lab}  eps={f['eps'][n]:+.2f}\nA(w) weight {f['w'][n]:.3f}", fontsize=9,
                         color="k" if f["w"][n] >= 1e-3 else "0.5")
            if fo > 1e-3:
                ax.set_facecolor("#f1f1f1"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        es = np.sort(f["eps"]); gap = es[nocc] - es[nocc - 1]
        pos = [e for e, ww in zip(f["eps"], f["w"]) if e > 1e-9 and ww >= 1e-3]; neg = [e for e, ww in zip(f["eps"], f["w"]) if e < -1e-9 and ww >= 1e-3]
        sg = (min(pos) - max(neg)) if pos and neg else float("nan")
        txt.set_text(f"{title}    U/t = {f['U']:.1f}    Z = {f['z']:.3f}\nqp gap (HOMO-LUMO) = {gap:.2f}     spectral gap = {sg:.2f}\n"
                     f"large circles joined left-right = the two states; small outer circles = their ghosts")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2500)); plt.close(fig)
    print("wrote", out, flush=True)


def critical_U(ucs, sgap, thr=0.01):
    """U/t where the spectral gap closes: linear extrapolation to zero of the last two grid points with gap > thr before the jump of the gap."""
    jump = next(i for i in range(1, len(sgap)) if sgap[i] > sgap[i - 1] + 0.5)
    ok = [i for i in range(jump) if sgap[i] > thr][-2:]
    (u1, g1), (u2, g2) = (ucs[ok[0]], sgap[ok[0]]), (ucs[ok[1]], sgap[ok[1]])
    return u2 - g2 * (u2 - u1) / (g2 - g1)


def make_density(fr, title, out, fps=12):
    Neff = fr[0]["Neff"]; N = 2 * Neff; nocc = N // 2; xy = skeleton_xy(Neff)
    cmap = plt.cm.viridis
    gaps = np.array([np.sort(f["eps"])[nocc] - np.sort(f["eps"])[nocc - 1] for f in fr])
    sgap = np.array([min([e for e, ww in zip(f["eps"], f["w"]) if e > 1e-9 and ww >= 1e-3] or [np.nan]) - max([e for e, ww in zip(f["eps"], f["w"]) if e < -1e-9 and ww >= 1e-3] or [np.nan]) for f in fr])
    rho12 = np.array([f["rho_star"][0, Neff] for f in fr])      # bond order center0-center1 (per spin)
    zs = np.array([f["z"] for f in fr]); dc = np.array([f["docc"] for f in fr]); ucs = np.array([f["U"] for f in fr])
    fig = plt.figure(figsize=(11, 5.4)); gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])
    ax0 = fig.add_subplot(gs[:, 0]); axa = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 1]); axc = fig.add_subplot(gs[2, 1])
    axa.plot(ucs, gaps, label="qp gap (HOMO-LUMO)"); axa.plot(ucs, sgap, label="spectral gap"); axa.legend(fontsize=8); axa.set_ylabel("gap")
    axb.plot(ucs, rho12, color="C2", label="bond order rho(c0,c1)"); axb.plot(ucs, zs, color="C3", label="Z"); axb.legend(fontsize=8)
    if np.isfinite(dc).any(): axc.plot(ucs, dc, color="C4", label="double occupancy"); axc.legend(fontsize=8)
    else: axc.plot(ucs, [f["rho_star"][0, 0] for f in fr], color="C0", label="state density (site 0)"); axc.plot(ucs, [f["rho_star"][fr[0]["Neff"], fr[0]["Neff"]] for f in fr], color="C1", ls="--", label="state density (site 1)"); axc.legend(fontsize=8)
    axc.set_xlabel("U/t")
    uc = critical_U(ucs, sgap)
    for ax in (axa, axb, axc): ax.axvline(uc, color="crimson", ls="--", lw=1.2)
    axa.text(uc - 0.15, axa.get_ylim()[1] * 0.50, f"$U_c/t$ = {uc:.2f} ", color="crimson", fontsize=9, va="top", ha="right")
    marks = [ax.axvline(ucs[0], color="k", lw=1) for ax in (axa, axb, axc)]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); fig.colorbar(sm, ax=ax0, fraction=0.04, label="qp density per spin  rho_aa (1 = full)")
    txt = fig.suptitle("")

    def update(i):
        f = fr[i]; ax0.clear()
        rho = np.diag(f["rho_star"]).copy()
        draw_sk(ax0, xy, Neff, None, color_by_sign=False, vals=np.clip(rho, 0, 1), bonds=None, cmap=cmap, scale=0.2)
        for s in range(2):
            ax0.text(xy[s * Neff, 0], xy[s * Neff, 1], f"{rho[s * Neff]:.2f}", ha="center", va="center", fontsize=9,
                     color="w" if rho[s * Neff] < 0.6 else "k", zorder=5)
        ax0.plot([xy[0, 0], xy[Neff, 0]], [xy[0, 1] + 0.0, xy[Neff, 1]], color="k", lw=0.3 + 12 * abs(rho12[i]), alpha=0.35, zorder=2)
        ax0.set_title(f"{title}   U/t = {f['U']:.1f}  (large circles = the two states, small outer = ghosts)\nbond order c0-c1 = {rho12[i]:.3f};  qp electrons per spin: {rho[:Neff].sum():.2f} | {rho[Neff:].sum():.2f}", fontsize=9)
        for m_ in marks: m_.set_xdata([f["U"], f["U"]])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
    fig.tight_layout()
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2500)); plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            f = line.split(); br2[round(float(f[0]), 1)] = f[2]
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "dV0"
    if which == "dV0":
        cases = {"scan_ng2_not_imp": (lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}", "Ng=2, dV=0"),
                 "scan_ng4_not_imp": (lambda U: f"scan_ng4_not_imp/U_{U:.1f}", "Ng=4, dV=0")}
    else:
        cases = {"scan_ng2_dV2_confirmed": (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", "Ng=2, dV=2"),
                 "scan_ng4_dV2_confirmed": (lambda U: f"scan_ng4_dV2_confirmed/U_{U:.1f}", "Ng=4, dV=2")}
    for d, (gd_, ttl) in cases.items():
        fr = load_frames(gd_)
        if len(sys.argv) <= 2 or sys.argv[2] != "density":
            make_orbitals(fr, ttl, f"{d}/qp_orbitals_vs_U.mp4")
        make_density(fr, ttl, f"{d}/qp_density_vs_U.mp4")
