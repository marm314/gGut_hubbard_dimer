"""(All videos are drawn in the chi basis of the solver: phi^qp_n = sum_i chi_i (C_qp)_{in}, C_qp = eigenvectors of H_qp as returned by eigh;
 the star basis chi' used for the orbital tracking is only an internal tool, see make_orbitals(key="ph").)
Videos of the quasiparticle orbitals of the Hubbard dimer (gGA, Delta V = 0) as U/t grows, drawn on the star skeleton of
$HOME/a.jpeg (a physical-like 'center' orbital per site with Ng satellite (ghost) orbitals attached to it; the two centers are linked).

Star basis of a site s: center c_s = R_s/|R_s| (the only qp orbital hopping to the other site, with -t |R_0||R_1|), and the satellites are the
eigenvectors of the on-site block H_s = -lambda_s projected on the orthogonal complement of c_s (sorted by energy), coupled to the center by v_a.
Half filling, as in gga_dimer.qp_state: the lowest N/2 of the N = 2(1+Ng) eigenvectors returned by eigh are filled (per spin), no fractional
occupation; the density is the code's own 1-RDM rho in the star basis.

orbitals video : one panel per eigenorbital of H_qp (top row occupied, bottom row virtual): circle area ~ |amplitude|^2, color = sign.
density video  : qp density rho_ii = sum_occ |(C_qp)_{in}|^2 of the chi_i (solver basis) on the skeleton positions, electrons per site, and gaps / Z / double occupancy vs U/t.
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
    R = np.loadtxt(sorted(glob.glob(os.path.join(d, "*_R.dat")))[0], ndmin=2)
    lam = np.loadtxt(sorted(glob.glob(os.path.join(d, "*_lambda.dat")))[0], ndmin=2)
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


def draw_sk(ax, xy, Neff, amp, color_by_sign=True, vals=None, scale=0.26, bonds=None, cmap=None, chi=False, hb=None, pw=None, labels=None):
    """chi=True: the chi_i of the solver basis (all the same size, no star bonds), labelled chi_1..chi_{1+Ng} of each site"""
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(xy[:, 0].min() - 0.45, xy[:, 0].max() + 0.45); ax.set_ylim(xy[:, 1].min() - 0.45, xy[:, 1].max() + 0.45)
    cen = (0, Neff)
    if chi and hb is not None:
        for k in range(2 * Neff):
            for l in range(k + 1, 2 * Neff):
                if abs(hb[k, l]) > (0.05 if k // Neff == l // Neff else 0.2): ax.plot([xy[k, 0], xy[l, 0]], [xy[k, 1], xy[l, 1]], color="0.55", lw=min(0.4 + 3.0 * abs(hb[k, l]), 4.0), alpha=0.6, zorder=1)
    for s in range(2):
        for a in range(1, Neff):
            if chi: break
            i, j = s * Neff, s * Neff + a
            lw = 0.6 if bonds is None else min(bonds[(i, j)], 3.5)
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="0.55", lw=lw, zorder=1)
    lw = 0.6 if bonds is None else min(bonds[(0, Neff)], 3.5)
    if not chi: ax.plot([xy[0, 0], xy[Neff, 0]], [xy[0, 1], xy[Neff, 1]], color="0.55", lw=lw, zorder=1)
    for k in range(2 * Neff):
        big = (k % Neff == 0) and not chi
        rad = scale * (1.35 if big else 1.0)
        ax.add_patch(Circle(xy[k], rad, fill=chi, fc="white", ec="0.75", lw=0.8, zorder=2))
        if color_by_sign:
            r = rad * np.sqrt(min(abs(amp[k]), 1.0))
            if r > 1e-3: ax.add_patch(Circle(xy[k], r, fc=POS if amp[k] > 0 else NEG, ec="none", zorder=3))
        else:
            ax.add_patch(Circle(xy[k], rad * 0.98, fc=cmap(vals[k]), ec="none", zorder=3))
        if labels is not None: ax.text(xy[k, 0], xy[k, 1] + rad + 0.11, labels[k], ha="center", va="center", fontsize=7, color="0.3")
        if chi and pw is not None: ax.add_patch(Circle(xy[k], rad * 1.08, fill=False, ec="k", lw=0.3 + 7.0 * pw[k], zorder=4))
        if chi: ax.text(xy[k, 0], xy[k, 1] + rad + 0.11, rf"$\chi_{{{k + 1}}}$", ha="center", va="center", fontsize=7, color="0.3")


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
        phi_raw = phi.copy()                            # raw eigh eigenvectors in the solver chi basis, energy order, no tracking, no rotation; sign only: largest component positive
        for n in range(dim):
            if phi_raw[np.argmax(np.abs(phi_raw[:, n])), n] < 0: phi_raw[:, n] *= -1
        raw = dict(eps=eps.copy(), occf=occf.copy(), w=w.copy(), phx=phi_raw, H=H.copy(), Rraw=[np.array(R[0], float), np.array(R[1], float)])
        line = open(d + "/output.out").read()
        z = float(re.search(r"Z0=([\d.eE+-]+)", line).group(1)); md = re.search(r"docc=([\d.eE+-]+)", line); docc = float(md.group(1)) if md else np.nan
        fr.append(dict(raw=raw, U=U, T=T, Rrow=np.eye(dim), rho_chi=rho0, rho_star=rho_star, occf=occf, eps=eps, ph=ph, w=w, info=info, tc=tc, z=z, docc=docc, Neff=Neff, R=R))
    align_frames(fr)
    for f in fr:     # eigenvectors in the solver (chi) basis, same orbitals/order/signs as the aligned star ones: ph_aligned = Rrow (T^T phi Col)  =>  phi Col = T Rrow^T ph_aligned
        f["phx"] = f["T"] @ f["Rrow"].T @ f["ph"]
        f["Rchi"] = [np.array(r_, dtype=float) for r_ in f["R"]]
    align_chi(fr)
    return fr


def align_chi(fr):
    """Relabelling of the chi_i. The solver returns every converged solution in its own basis of the chi_i inside each site block. Here the chi_i of a site are only PERMUTED (and
    sign-flipped) between consecutive U/t, no mixing: the permutation P_s (Hungarian on the overlaps of the block rows of the tracked orbitals with the previous U/t) and the signs
    are applied to R_s, to the block rows of the eigenvectors and to rho (energies, occupations and C_{R_s,n} do not change). Propagation order as align_frames."""
    from scipy.optimize import linear_sum_assignment
    Neff = fr[0]["Neff"]
    if Neff == 1: return
    ref = int(np.argmax([np.diff(np.sort(f["eps"])).min() for f in fr]))
    pairs = [(fr[k - 1], fr[k]) for k in range(ref + 1, len(fr))] + [(fr[k + 1], fr[k]) for k in range(ref - 1, -1, -1)]
    for p_, c_ in pairs:
        c_["rho_chi"] = c_["rho_chi"].copy()
        for s_ in range(2):
            rows = slice(s_ * Neff, (s_ + 1) * Neff)
            M = p_["phx"][rows, :] @ c_["phx"][rows, :].T                  # overlap of row i (previous) with row j (current)
            ri, ci = linear_sum_assignment(-np.abs(M))
            Q = np.zeros((Neff, Neff))
            for i_, j_ in zip(ri, ci): Q[i_, j_] = 1.0 if M[i_, j_] >= 0 else -1.0
            c_["phx"][rows, :] = Q @ c_["phx"][rows, :]; c_["Rchi"][s_] = Q @ c_["Rchi"][s_]
            c_["rho_chi"][rows, :] = Q @ c_["rho_chi"][rows, :]; c_["rho_chi"][:, rows] = c_["rho_chi"][:, rows] @ Q.T


SAT_DEG_TOL = 0.05      # satellites of a site closer than this (in t) are treated as degenerate
DEG_TOL = 0.06          # levels closer than this (in t) are treated as one degenerate group when aligning the orbitals


def align_frames(fr):
    """Make the orbitals continuous in U/t. (i) Satellites: the star satellites of a site are re-identified between consecutive U/t
    (Hungarian on |d e| + |d v|) so a satellite keeps its position on the skeleton; the eigenvector components are permuted accordingly.
    (ii) Signs: maximum-overlap partner in the previous frame, positive overlap (panels stay in energy order)."""
    from scipy.optimize import linear_sum_assignment
    Neff = fr[0]["Neff"]; Ng = Neff - 1
    # Start from the frame where the levels are best separated (largest minimum gap between neighbouring levels): there the orbitals are canonical eigenstates. Where the levels are
    # (nearly) degenerate eigh returns an arbitrary basis, so these frames must inherit their orbitals from the well-defined ones, not the other way round: propagate outwards from
    # the reference frame, upwards and downwards in U/t.
    ref = int(np.argmax([np.diff(np.sort(f["eps"])).min() for f in fr]))
    pairs = [(fr[k - 1], fr[k]) for k in range(ref + 1, len(fr))] + [(fr[k + 1], fr[k]) for k in range(ref - 1, -1, -1)]
    for p, c_ in pairs:
        # satellite identity between consecutive frames: among all assignments of the satellites of each site (Ng! per site), take the one that makes the orbitals most continuous
        # (largest sum of the maximum-overlap assignment between the previous and the current orbitals); cheaper and more robust than matching satellite energies.
        import itertools
        best_sc, perm = -1.0, np.arange(2 * Neff)
        for q0 in itertools.permutations(range(Ng)):
            for q1 in itertools.permutations(range(Ng)):
                pm = np.arange(2 * Neff)
                for a_ in range(Ng): pm[1 + a_] = 1 + q0[a_]; pm[Neff + 1 + a_] = Neff + 1 + q1[a_]
                ovc = np.abs(p["ph"].T @ c_["ph"][pm, :]); r_, c2_ = linear_sum_assignment(-ovc); sc_ = ovc[r_, c2_].sum()
                if sc_ > best_sc + 1e-12: best_sc, perm = sc_, pm
        # new position i of a component = old index perm[i]
        c_["ph"] = c_["ph"][perm, :]; c_["Rrow"] = c_["Rrow"][perm, :]
        c_["info"] = [(c_["info"][s_][0], c_["info"][s_][1][perm[s_ * Neff + 1:(s_ + 1) * Neff] - s_ * Neff - 1],
                       c_["info"][s_][2][perm[s_ * Neff + 1:(s_ + 1) * Neff] - s_ * Neff - 1]) for s_ in range(2)]
        # (ii b) degenerate satellites: if the satellites of a site have (nearly) the same energy, their basis inside that subspace is arbitrary (eigh gives a different rotation
        # at every U/t). Rotate them (orthogonal Procrustes, one rotation per site) so that the satellite components of the orbitals stay as close as possible to the previous frame;
        # the coupling vector to the centre (used for the bonds) is rotated with them.
        if Ng >= 2:
            for s_ in range(2):
                if np.ptp(c_["info"][s_][1]) < SAT_DEG_TOL:
                    rows = list(range(s_ * Neff + 1, (s_ + 1) * Neff)); A_s = p["ph"][rows, :]; B_s = c_["ph"][rows, :]
                    Uu, _, Vt = np.linalg.svd(A_s @ B_s.T); Rot = Uu @ Vt
                    c_["ph"][rows, :] = Rot @ B_s; c_["Rrow"][rows, :] = Rot @ c_["Rrow"][rows, :]
                    c_["info"][s_] = (c_["info"][s_][0], c_["info"][s_][1], Rot @ c_["info"][s_][2])
        # (iii) orbital tracking. Each panel follows one orbital: the current orbitals are matched to the previous frame's by maximum overlap (Hungarian on |<prev|cur>|) and
        # permuted into the previous slots (eps, w, occupation move with them), so an orbital never jumps to another panel when levels cross or split.
        # Near-degenerate levels (closer than DEG_TOL): eigh returns an arbitrary basis of the subspace, so the orbitals of each such group are first rotated
        # (orthogonal Procrustes) onto the previous orbitals they overlap most with.
        N_ = c_["ph"].shape[1]; e_ = c_["eps"]; g0 = 0
        for n in range(1, N_ + 1):
            if n == N_ or e_[n] - e_[n - 1] > DEG_TOL:
                if n - g0 >= 2:
                    G = list(range(g0, n)); M = c_["ph"][:, G]
                    S_ = p["ph"].T @ M                                  # <prev_slot | cur_member>
                    rr, cc2 = linear_sum_assignment(-np.abs(S_.T))      # member r <-> previous slot cc2[r]
                    P = p["ph"][:, cc2]; Uu, _, Vt = np.linalg.svd(M.T @ P); c_["ph"][:, G] = M @ (Uu @ Vt)
                g0 = n
        ov = p["ph"].T @ c_["ph"]                                       # <prev_slot m | cur_n>
        slot, cur = linear_sum_assignment(-np.abs(ov))                  # previous slot <-> current orbital
        order = np.empty(N_, dtype=int); order[slot] = cur              # new column m = current orbital order[m]
        for key in ("eps", "w", "occf"): c_[key] = c_[key][order]
        c_["ph"] = c_["ph"][:, order]
        # phases. Two kinds of sign freedom, both fixed against the previous frame: the sign of each orbital (column), and the sign of each satellite basis vector of the star basis
        # (row; eigh/the star construction may flip it from frame to frame, which would recolour that ghost circle in every orbital at once). Alternate: row signs from the
        # current column signs, column signs from the current row signs (the centre rows are not flipped: global sign fixed by the columns).
        A_ = c_["ph"]; B_ = p["ph"]; sat = [r for r in range(A_.shape[0]) if r % Neff != 0]
        sig = np.array([1.0 if B_[:, n] @ A_[:, n] >= 0 else -1.0 for n in range(N_)]); rs = np.ones(A_.shape[0])
        for _ in range(6):
            for r in sat:
                v = float((A_[r, :] * sig) @ B_[r, :]); rs[r] = 1.0 if v >= 0 else -1.0
            for n in range(N_):
                v = float((rs * A_[:, n]) @ B_[:, n]); sig[n] = 1.0 if v >= 0 else -1.0
        c_["ph"] = (A_ * rs[:, None]) * sig[None, :]; c_["Rrow"] = rs[:, None] * c_["Rrow"]


def perm_only_frames(fr):
    """PERMUTATIONS ONLY (the alignment used by all the chi-basis movies). Raw eigenvectors of H_qp in the solver chi basis, in the energy order of eigh (the qp states are never permuted, never
    mixed, no orbital tracking); the only operations: (i) the chi_i of each site are reordered (as if the AO basis of the site were listed in another order) at every U/t to follow the previous
    U/t, by exhaustive search over the permutations of each site (sign-sensitive match sum_n |<A_n|X_n>|, A = previous aligned frame); (ii) the overall sign of each qp state. The same
    permutation is applied to R_s (Rchi) and to rho (rho_chi). Frames: energy order, eps/occf/w/phx of the raw eigh."""
    import itertools
    rf = [dict(f, **{k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in f["raw"].items()}) for f in fr]
    Neff = rf[0]["Neff"]; N = 2 * Neff
    for f in rf: f["Rchi"] = [np.array(r_, dtype=float) for r_ in f["R"]]
    if Neff == 1: return rf
    ref = int(np.argmax([np.diff(np.sort(f["eps"])).min() for f in rf]))
    pairs = [(rf[k - 1], rf[k]) for k in range(ref + 1, len(rf))] + [(rf[k + 1], rf[k]) for k in range(ref - 1, -1, -1)]
    P = np.array(list(itertools.permutations(range(Neff))))                   # (n!, Neff)
    for p_, c_ in pairs:
        X0 = c_["phx"].copy(); A = p_["phx"]; Cs = []
        for s_ in range(2):
            rows = np.arange(s_ * Neff, (s_ + 1) * Neff)
            Cs.append(np.einsum("in,pin->pn", A[rows, :], X0[rows[P], :]))      # (n!, N): site-s part of <A_n|X_n> for every permutation of the chi_i of site s
        score = np.abs(Cs[0][:, None, :] + Cs[1][None, :, :]).sum(axis=2)       # (n!, n!)
        i0, i1 = np.unravel_index(np.argmax(score), score.shape)
        idx = np.concatenate([P[i0], Neff + P[i1]])                              # new chi_i = old chi_idx[i]
        X = X0[idx, :]
        sg = np.where(np.sum(A * X, axis=0) < 0, -1.0, 1.0); X = X * sg
        c_["phx"] = X
        for s_ in range(2): c_["Rchi"][s_] = c_["Rchi"][s_][idx[s_ * Neff:(s_ + 1) * Neff] - s_ * Neff]
        c_["rho_chi"] = c_["rho_chi"][np.ix_(idx, idx)]
    return rf


def bonds_from(f):
    Neff = f["Neff"]; b = {}
    scale = 4.0
    for s in range(2):
        v = f["info"][s][2]
        for a in range(1, Neff): b[(s * Neff, s * Neff + a)] = max(0.3, scale * abs(v[a - 1]))
    b[(0, Neff)] = max(0.3, scale * abs(f["tc"]))
    return b


def make_orbitals(fr, title, out, fps=12, key="phx", mode="perm"):
    """key='ph': the original star basis chi' (chi'_1 = state = direction of R_s, others ghosts), orbitals followed in U/t (tracked); key='phx': solver chi, permutations only"""
    """key='phx': eigenvectors phi^qp_n = sum_i chi_i (C_qp)_{in} in the solver chi basis; key='ph': star basis chi'"""
    if key == "ph": mode = None          # chi' movies: orbitals followed in U/t (caption of the tracked version)
    Neff = fr[0]["Neff"]; N = 2 * Neff; xy = skeleton_xy(Neff); nocc = N // 2
    fig, axs = plt.subplots(2, nocc, figsize=(3.4 * nocc, 6.2), squeeze=False)
    txt = fig.suptitle("", fontsize=12)

    def update(i):
        f = fr[i]
        for n in range(N):
            ax = axs[n // nocc, n % nocc]; ax.clear()
            v_ = f.get(key, f["ph"])[:, n]; amp = v_ ** 2 * np.sign(v_)
            hb_ = None; pw_ = None      # bonds (|H_qp,ij|) and physical-weight rings (R_si^2/|R_s|^2) were tried and removed: too busy
            if hb_ is not None: pw_ = np.concatenate([np.asarray(f["Rraw"][s_]) ** 2 / np.sum(np.asarray(f["Rraw"][s_]) ** 2) for s_ in range(2)])
            draw_sk(ax, xy, Neff, amp, bonds=None if key == "phx" else bonds_from(f), chi=(key == "phx"), hb=hb_, pw=pw_, labels=(None if key == "phx" else [rf"$\chi'_{{{k_ + 1}}}$" for k_ in range(2 * Neff)]))
            fo = f["occf"][n]
            lab = "occ" if fo > 0.5 else "virt"
            ax.set_title(f"{lab}  eps={f['eps'][n]:+.2f}\nA(w) weight {f['w'][n]:.3f}", fontsize=9,
                         color="k" if f["w"][n] >= 1e-3 else "0.5")
            if fo > 1e-3:
                ax.set_facecolor("#f1f1f1"); ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        es = np.sort(f["eps"]); gap = es[nocc] - es[nocc - 1]
        pos = [e for e, ww in zip(f["eps"], f["w"]) if e > 1e-9 and ww >= 1e-3]; neg = [e for e, ww in zip(f["eps"], f["w"]) if e < -1e-9 and ww >= 1e-3]
        sg = (min(pos) - max(neg)) if pos and neg else float("nan")
        txt.set_text(f"{title}    U/t = {f['U']:.1f}    Z = {f['z']:.3f}\nqp gap (HOMO-LUMO) = {gap:.2f}     spectral gap = {sg:.2f}\n"
                     + (r"$\phi^{qp}_n=\sum_i\chi_i(C_{qp})_{in}$: circles = $\chi_i$, $i=1..2(1+N_g)$: site 1 (left), then site 2 (right)" + ("; raw solver basis" if mode == "raw" else ("\n$\\chi$ = solver basis, only the order of the $\\chi_i$ within each site is changed" if mode == "perm" else "\n$\\chi$ = solver basis, $\\chi_i$ permuted/sign-flipped per site to follow the previous $U/t$")) if key == "phx" else r"$\phi^{qp}_n=\sum_j\chi'_j(C'_{qp})_{jn}$: $\chi'_1,\chi'_{%d}$ = the two states (direction of $R_s$), small circles = their ghosts" % (Neff + 1)) + ("\neigh order, no tracking, no rotation; sign: largest component > 0" if mode == "raw" else ("\neigh order, no orbital tracking, no rotation; orbital signs aligned to the previous $U/t$" if mode == "perm" else "\neach panel follows one orbital (matched to the previous U/t; phases aligned)")))
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2500)); plt.close(fig)
    print("wrote", out, flush=True)


def critical_U(ucs, sgap, thr=0.01):
    """U/t where the spectral gap closes: linear extrapolation to zero of the last two grid points with gap > thr before the jump of the gap."""
    jump = next(i for i in range(1, len(sgap)) if sgap[i] > sgap[i - 1] + 0.5)
    ok = [i for i in range(jump) if sgap[i] > thr][-2:]
    (u1, g1), (u2, g2) = (ucs[ok[0]], sgap[ok[0]]), (ucs[ok[1]], sgap[ok[1]])
    return u2 - g2 * (u2 - u1) / (g2 - g1)


def make_density(fr, title, out, fps=12, basis="chi"):
    """basis='chi': rho_ii in the (permuted) solver chi basis; basis='chip': the original star basis chi' (state chi'_1 of each site + ghosts), with the bond order between the two states"""
    chip = basis == "chip"
    Neff = fr[0]["Neff"]; N = 2 * Neff; nocc = N // 2; xy = skeleton_xy(Neff)
    cmap = plt.cm.viridis
    gaps = np.array([np.sort(f["eps"])[nocc] - np.sort(f["eps"])[nocc - 1] for f in fr])
    sgap = np.array([min([e for e, ww in zip(f["eps"], f["w"]) if e > 1e-9 and ww >= 1e-3] or [np.nan]) - max([e for e, ww in zip(f["eps"], f["w"]) if e < -1e-9 and ww >= 1e-3] or [np.nan]) for f in fr])
    if chip: rho12 = np.array([f["rho_star"][0, Neff] for f in fr])      # bond order between the two states chi'_1 of the sites (per spin)
    else: rho12 = np.array([np.trace(f.get("rho_chi", f["rho_star"])[:Neff, :Neff]) for f in fr])      # qp electrons per spin on the chi_i of site 1 (basis-independent inside the block)
    zs = np.array([f["z"] for f in fr]); dc = np.array([f["docc"] for f in fr]); ucs = np.array([f["U"] for f in fr])
    fig = plt.figure(figsize=(11, 5.4)); gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])
    ax0 = fig.add_subplot(gs[:, 0]); axa = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 1]); axc = fig.add_subplot(gs[2, 1])
    axa.plot(ucs, gaps, label="qp gap (HOMO-LUMO)"); axa.plot(ucs, sgap, label="spectral gap"); axa.legend(fontsize=8); axa.set_ylabel("gap")
    axb.plot(ucs, rho12, color="C2", label=("bond order rho(chi'_1,chi'_4)" if chip else "qp electrons/spin on site 1")); axb.plot(ucs, zs, color="C3", label="Z"); axb.legend(fontsize=8)
    if np.isfinite(dc).any(): axc.plot(ucs, dc, color="C4", label="double occupancy"); axc.legend(fontsize=8)
    else: axc.plot(ucs, [f["rho_star"][0, 0] for f in fr], color="C0", label="state density (site 0)"); axc.plot(ucs, [f["rho_star"][fr[0]["Neff"], fr[0]["Neff"]] for f in fr], color="C1", ls="--", label="state density (site 1)"); axc.legend(fontsize=8)
    axc.set_xlabel("U/t")
    uc = critical_U(ucs, sgap)
    for ax in (axa, axb, axc): ax.axvline(uc, color="crimson", ls="--", lw=1.2)
    axa.text(uc - 0.15, axa.get_ylim()[1] * 0.50, f"$U_c/t$ = {uc:.2f} ", color="crimson", fontsize=9, va="top", ha="right")
    marks = [ax.axvline(ucs[0], color="k", lw=1) for ax in (axa, axb, axc)]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); fig.colorbar(sm, ax=ax0, fraction=0.04, label=(r"qp density per spin  $\rho_{jj}$ in $\chi'$ (1 = full)" if chip else r"qp density per spin  $\rho_{ii}$ (1 = full)"))
    txt = fig.suptitle("")

    def update(i):
        f = fr[i]; ax0.clear()
        if chip:
            rho = np.diag(f["rho_star"]).copy()
            draw_sk(ax0, xy, Neff, None, color_by_sign=False, vals=np.clip(rho, 0, 1), bonds=None, cmap=cmap, scale=0.2, labels=[rf"$\chi'_{{{k_ + 1}}}$" for k_ in range(2 * Neff)])
            for s in range(2):
                ax0.text(xy[s * Neff, 0], xy[s * Neff, 1], f"{rho[s * Neff]:.2f}", ha="center", va="center", fontsize=9, color="w" if rho[s * Neff] < 0.6 else "k", zorder=5)
            ax0.plot([xy[0, 0], xy[Neff, 0]], [xy[0, 1] + 0.0, xy[Neff, 1]], color="k", lw=0.3 + 12 * abs(rho12[i]), alpha=0.35, zorder=2)
            ax0.set_title(rf"{title}   U/t = {f['U']:.1f}   $\rho_{{jj}}$ in the $\chi'$ basis" + "\n" + rf"large circles = states $\chi'_1,\chi'_{{{Neff + 1}}}$, small = ghosts;  bond order between the states = {rho12[i]:.3f}" + f"\nqp electrons per spin: {rho[:Neff].sum():.2f} | {rho[Neff:].sum():.2f}", fontsize=9)
        else:
            rho = np.diag(f.get("rho_chi", f["rho_star"])).copy()
            draw_sk(ax0, xy, Neff, None, color_by_sign=False, vals=np.clip(rho, 0, 1), bonds=None, cmap=cmap, scale=0.2, chi=True)
            for k_ in range(2 * Neff):
                ax0.text(xy[k_, 0], xy[k_, 1], f"{rho[k_]:.2f}", ha="center", va="center", fontsize=8, color="w" if rho[k_] < 0.6 else "k", zorder=5)
            ax0.set_title(rf"{title}   U/t = {f['U']:.1f}   $\rho_{{ii}}$ in the $\chi_i$ basis (circles = $\chi_i$ of site 1 left, site 2 right)" + f"\nqp electrons per spin: {rho[:Neff].sum():.2f} | {rho[Neff:].sum():.2f}", fontsize=9)
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
        cases = {"scan_ng2_not_imp": (lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}", "Ng=2, dV=0")}
    else:
        cases = {"scan_ng2_dV2_confirmed": (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", "Ng=2, dV=2")}
    for d, (gd_, ttl) in cases.items():              # (Ng=4 is not touched any more)
        fr = perm_only_frames(load_frames(gd_))      # permutations of the chi_i inside each site + sign of each qp state; the qp states stay in energy order
        if len(sys.argv) <= 2 or sys.argv[2] != "density":
            make_orbitals(fr, ttl, f"{d}/qp_orbitals_vs_U.mp4", mode="perm")
        make_density(fr, ttl, f"{d}/qp_density_vs_U.mp4")
