"""Videos of the spectral weight carried by each quasiparticle level as U/t increases: for every U/t,
the eigenvalues eps_n of H_qp (x axis) with their weights w_n = 2 sum_sites |Rmat^T phi_n|^2 (stems; same
normalization as A(omega) of gga_dimer.compute_spectral_function, so sum_n w_n = A integrated), and the exact (ED)
poles as red circles. Top: linear scale; bottom: log scale (shows the weightless levels, e.g. sleeping ghosts).

Requires ffmpeg; writes <scan_dir>/qp_weights_vs_U.mp4."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plot_eqp_3d import load_R_lambda, t, OMEGA_MAX
import plot_eqp_3d_dV2 as q

Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]
PAL = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
PAL_NAMES = ["blue", "orange", "green", "purple", "brown"]


WTHR = 1e-3   # a level is a "peak" if its weight is at least this
GATE = 1.0    # max |d eps| + |d w| between consecutive U points for a peak to keep its identity
BIG = 0.05    # an unmatched peak heavier than this ends a continuity segment


def track_colors(frames):
    """Persistent peak identity along U, colored by energy order.
    1) Peaks = levels with weight >= WTHR. On the positive side, peaks of consecutive U points are matched one to one (Hungarian
       assignment on |d eps| + |d w|; matches above GATE are rejected), so a peak keeps its identity while its weight and energy change;
       unmatched peaks are new identities. A heavy (> BIG) unmatched peak ends a continuity segment (discontinuous change of the solution).
    2) In each segment, the identities present in the frame with most peaks are colored blue, orange, green, ... in order of increasing
       energy there; the other identities of the segment take the lowest color not used by an identity coexisting with them.
    Levels without weight are light gray; negative-energy peaks take the color of the positive peak of the same frame with the closest
    |eps| and weight."""
    from scipy.optimize import linear_sum_assignment
    n = len(frames); ids_f = []; prev = []; nid = 0; seg = [0] * n
    for k, fr_ in enumerate(frames):
        eps, w = fr_[0], fr_[1]
        pk = sorted([i for i in range(len(eps)) if eps[i] >= -1e-9 and w[i] >= WTHR], key=lambda i: eps[i])
        cur = {}
        if prev and pk:
            cost = np.array([[abs(eps[i] - p[0]) + abs(w[i] - p[1]) for p in prev] for i in pk])
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri, ci] < GATE: cur[pk[ri]] = prev[ci][2]
        matched_prev = set(cur.values())
        broke = any(p[1] > BIG and p[2] not in matched_prev for p in prev)
        for i in pk:
            if i not in cur:
                cur[i] = nid; nid += 1
                if w[i] > BIG and prev: broke = True
        seg[k] = seg[k - 1] + (1 if (broke and k > 0) else 0) if k > 0 else 0
        ids_f.append(cur); prev = [(eps[i], w[i], cur[i]) for i in pk]
    # color indices per (segment, identity)
    cidx = {}
    for sg in sorted(set(seg)):
        ks = [k for k in range(n) if seg[k] == sg]
        kref = max(ks, key=lambda k: (len(ids_f[k]), -k))
        eps_ref = frames[kref][0]
        order = sorted(ids_f[kref], key=lambda i: eps_ref[i])
        for r, i in enumerate(order): cidx[ids_f[kref][i]] = r
        others = []
        for k in ks:
            for i, ident in ids_f[k].items():
                if ident not in cidx and ident not in others: others.append(ident)
        for ident in others:
            used = {cidx[o] for k in ks if ident in ids_f[k].values() for o in ids_f[k].values() if o in cidx}
            cidx[ident] = next(q for q in range(100) if q not in used)
    out = []
    for k, fr_ in enumerate(frames):
        eps, w = fr_[0], fr_[1]
        cols = ["lightgray"] * len(eps)
        for i, ident in ids_f[k].items(): cols[i] = PAL[min(cidx[ident], len(PAL) - 1)]
        pk = list(ids_f[k])
        for i in range(len(eps)):
            if eps[i] < -1e-9 and w[i] >= WTHR and pk:
                j = min(pk, key=lambda j: abs(abs(eps[i]) - eps[j]) + abs(w[i] - w[j]))
                cols[i] = cols[j]
        out.append(cols)
    return out


def match_to_ref(frames, ref_frames, ref_cols, gate=1.5):
    """Colors of a larger-N_g scan from the N_g=2 scan of the same Delta V: at each U/t the positive peaks are matched one to one
    to the N_g=2 peaks (Hungarian on |d eps| + |d w|, gate `gate`) and take their colors, so the same physical peak has the same color
    at N_g=2 and N_g=4; unmatched peaks (the extra ones) are purple and brown in order of increasing energy."""
    from scipy.optimize import linear_sum_assignment
    out = []
    for k, fr_ in enumerate(frames):
        eps, w = fr_[0], fr_[1]; re_, rw = ref_frames[k][0], ref_frames[k][1]
        pk = sorted([i for i in range(len(eps)) if eps[i] >= -1e-9 and w[i] >= WTHR], key=lambda i: eps[i])
        rp = [j for j in range(len(re_)) if re_[j] >= -1e-9 and rw[j] >= WTHR]
        cols = ["lightgray"] * len(eps); matched = set()
        if pk and rp:
            cost = np.array([[abs(eps[i] - re_[j]) + abs(w[i] - rw[j]) for j in rp] for i in pk])
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri, ci] < gate: cols[pk[ri]] = ref_cols[k][rp[ci]]; matched.add(pk[ri])
        extra = [i for i in pk if i not in matched]
        for q, i in enumerate(extra): cols[i] = PAL[3 + min(q, 1)]
        for i in range(len(eps)):
            if eps[i] < -1e-9 and w[i] >= WTHR and pk:
                j = min(pk, key=lambda j: abs(abs(eps[i]) - eps[j]) + abs(w[i] - w[j]))
                cols[i] = cols[j]
        out.append(cols)
    return out


def qp_levels(R, lam):
    R0, R1 = R; lam0, lam1 = lam; Neff = len(R0); dim = 2 * Neff
    H = np.zeros((dim, dim))
    H[:Neff, :Neff] = -lam0; H[Neff:, Neff:] = -lam1
    H[:Neff, Neff:] = -t * np.outer(R0, R1); H[Neff:, :Neff] = -t * np.outer(R1, R0)
    Rm = np.zeros((dim, 2)); Rm[:Neff, 0] = R0; Rm[Neff:, 1] = R1
    eps, phi = np.linalg.eigh(H)
    w = 2.0 * ((Rm.T @ phi) ** 2).sum(axis=0)
    return eps, w


def ed_poles0(U):
    Om = np.sqrt(U ** 2 + 16 * t ** 2)
    p = [s * (Om / 2 - t) for s in (1, -1)] + [s * (Om / 2 + t) for s in (1, -1)]
    w = [1 + 4 * t / Om] * 2 + [1 - 4 * t / Om] * 2
    return np.array(p), np.array(w)


def ed_poles2(U):
    p, w = q.ed_dV.analytic_poles(t, U, q.DELTA_V_ED)
    return np.array(p), np.array(w)


def make_video(get_dir, ed_fn, title, out, fps=12, ref_dir=None):
    fr = []
    for U in Us:
        R, lam = load_R_lambda(get_dir(U))
        eps, w = qp_levels(R, lam); pe, we = ed_fn(U); fr.append((eps, w, pe, we))
    if ref_dir is None:
        ids = track_colors(fr)
    else:
        fr2 = [qp_levels(*load_R_lambda(ref_dir(U))) for U in Us]
        ids = match_to_ref(fr, fr2, track_colors(fr2))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.8), sharex=True)
    a1.set_ylim(0, 2.3); a2.set_yscale("log"); a2.set_ylim(1e-14, 5)
    a2.set_xlim(-OMEGA_MAX, OMEGA_MAX)
    a2.set_xlabel("quasiparticle energy / t"); a1.set_ylabel("level weight"); a2.set_ylabel("level weight (log)")
    a1.set_title(title); txt = a1.text(0.03, 0.88, "", transform=a1.transAxes, fontsize=13)
    arts = []

    def update(i):
        nonlocal arts
        for a in arts: a.remove()
        eps, w, pe, we = fr[i]; arts = []
        cols = ids[i]
        for ax in (a1, a2):
            arts.append(ax.vlines(eps, 1e-14 if ax is a2 else 0, np.maximum(w, 1e-13), colors=cols, lw=3))
            arts.append(ax.scatter(eps, np.maximum(w, 1e-13), s=28, color=cols, zorder=3))
            arts.append(ax.scatter(pe, we, s=70, facecolors="none", edgecolors="red", lw=1.4, zorder=4))
        txt.set_text(f"U/t = {Us[i]:.1f}")
        return arts

    nlev = len(fr[0][0]) // 2
    a1.legend(handles=[plt.Line2D([], [], color=PAL[j], marker="o", lw=3, label=f"{PAL_NAMES[j]}")
                       for j in range(nlev)] +
                      [plt.Line2D([], [], color="lightgray", marker="o", lw=3, label=f"no weight (< {WTHR:g})"),
                       plt.Line2D([], [], color="red", marker="o", mfc="none", ls="", ms=8, label="ED pole")],
              loc="upper right", fontsize=8, title="gGA peak (identity tracked in U)", title_fontsize=8)
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(Us), blit=False)
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800)); plt.close(fig)
    print(f"wrote {out} ({len(Us)} frames)", flush=True)


if __name__ == "__main__":
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            f = line.split(); br2[round(float(f[0]), 1)] = f[2]
    make_video(lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}", ed_poles0, r"Ng=2, $\Delta V=0$", "scan_ng2_not_imp/qp_weights_vs_U.mp4")
    make_video(lambda U: f"scan_ng4_not_imp/U_{U:.1f}", ed_poles0, r"Ng=4, $\Delta V=0$", "scan_ng4_not_imp/qp_weights_vs_U.mp4",
               ref_dir=lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}")
    make_video(lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", ed_poles2, r"Ng=2, $\Delta V=2$", "scan_ng2_dV2_confirmed/qp_weights_vs_U.mp4")
    make_video(lambda U: f"scan_ng4_dV2_confirmed/U_{U:.1f}", ed_poles2, r"Ng=4, $\Delta V=2$", "scan_ng4_dV2_confirmed/qp_weights_vs_U.mp4",
               ref_dir=lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}")
