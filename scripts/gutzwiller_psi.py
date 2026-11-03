"""Explicit Gutzwiller wave function |Psi> = P_0 P_1 |Psi0> of a converged gGA solution of the half-filled dimer,
built in the physical 2-site Fock space (16 states; the N_up=N_dn=1 sector has 6) so that its ionic / covalent content,
energy and hole spectral weights can be read off directly (gGA itself never forms |Psi>: it only uses local quantities).

Construction (per fragment i, Neff = 1+Ng quasiparticle orbitals per spin, 2*Neff qp modes ordered [a0 up, a0 dn, a1 up, ...]):
  |Psi0>  Slater determinant: lowest Neff levels of H_qp filled for each spin (the code's rule, gga_dimer.qp_state),
  M       its amplitudes over the qp Fock basis |n0>|n1> (JW basis, site 0 modes first), shape (4^Neff, 4^Neff),
  P0_i    local qp density matrix of Psi0 on fragment i: P0_0 = M M^T, P0_1 = M^T M,
  Phi_i   impurity ground state (physical orbital + Neff bath orbitals) as a matrix Phi[Gamma, nbar]  (Gamma = 4 physical
          configurations, nbar = bath configurations),
  Phit_i  = Phi_i U_PH: the bath particle-hole transformed back to the qp Fock space, U_PH f^dag_j U_PH^dag = f_j,
          U_PH|0> = |full> (so that Delta = 1 - <d^dag d> and D = <d^dag c> become <f^dag f> and Tr[Phi^dag c^dag Phi f]),
  lambda_i = Phit_i P0_i^(-1/2)   (pseudo-inverse on the support of P0_i),
  Psi[G0,G1] = sum_{n0,n1} lambda_0[G0,n0] M[n0,n1] lambda_1[G1,n1]   (physical Fock vector, index 4*G0+G1, modes 0u,0d,1u,1d).
Checks printed: (i) R recomputed from Phit (R = S^-1 Tr[Phit^T c^dag Phit f_a]) against the R of the solution (tests the particle-hole
phase convention, see ph_phase); (ii) the norm <Psi|Psi> (equal to 1 only in the Gutzwiller limit); (iii) the weight of Phit in the
null space of P0 (inert ghosts); (iv) the residual of the embedding state.

Works from both solver families: a directory with z.npy (non-impurity solver, not_imp_gga_dimer.py) supplies phi_i itself; otherwise
(impurity solver, gga_dimer.run_gga) phi_i is the ground state of H_imp rebuilt from (R, lambda) as in one iteration of the loop.
The exact reference uses Delta V_ED = -2 dV (gGA lowers site 0, exact_dimer_deltaV raises it).

Usage:  python gutzwiller_psi.py dV0 2|4 [U ...]     (scan_ng2_not_imp / scan_ng4_not_imp, as ionic_covalent.py)
        python gutzwiller_psi.py dV2 2|4 [U ...]     (scan_ng{2,4}_dV2_confirmed)
        DV=<dV> python gutzwiller_psi.py <dir> [U ...]   (any directory with U_<U>/ subdirectories)
"""
import os
import sys
import numpy as np
import gga_dimer as gg
import exact_dimer_deltaV as ed
from make_qp_orbital_videos import load_R_lambda

t = 1.0
SHIFT = 1e-9


# ------------------------------------------------------------------ embedding state of one fragment
def impurity_gs(V, lam_c, ops):
    """Ground state vector of H_imp, same construction as gga_dimer.impurity_solve (which returns only expectation values)."""
    Neff = ops["Neff"]
    H = ops["H_loc"]
    for a in range(Neff):
        H = H + V[a] * (ops["hyb_up"][a] + ops["hyb_dn"][a])
        for b in range(Neff):
            if lam_c[a, b] != 0.0:
                H = H + lam_c[a, b] * (ops["lam_up"][a][b] + ops["lam_dn"][a][b])
    sec = ops["sector"]
    w, v = np.linalg.eigh(H.tocsr()[sec][:, sec].toarray())
    gs = np.zeros(H.shape[0]); gs[sec] = v[:, 0]
    return gs, w[:2]


def embedding_vector(d, i, ops, Neff, V=None, lam_c=None):
    """Embedding state of fragment i. Non-impurity solution (z.npy = [phi_0, phi_1, X], phi_i on the N_up=N_dn sector of the impurity
    Fock space): the saved phi_i itself. Otherwise (impurity solver): the ground state of H_imp built from (V, lambda^c).
    Returns (vector, residual |H phi - E phi| (None if V, lambda^c are not available), number of sector eigenstates below it, label)."""
    sec = ops["sector"]; m = len(sec)
    zf = os.path.join(d, "z.npy")
    z = np.load(zf) if os.path.exists(zf) else None
    if z is not None and len(z) == 2 * m + 2 * Neff * Neff:
        phi = z[i * m:(i + 1) * m]; phi = phi / np.linalg.norm(phi)
        gs = np.zeros(2 ** (2 * (1 + Neff))); gs[sec] = phi
        resid = rank = None
        if V is not None:
            H = ops["H_loc"]
            for a in range(Neff):
                H = H + V[a] * (ops["hyb_up"][a] + ops["hyb_dn"][a])
                for b in range(Neff):
                    if lam_c[a, b] != 0.0:
                        H = H + lam_c[a, b] * (ops["lam_up"][a][b] + ops["lam_dn"][a][b])
            Hg = H @ gs; E = gs @ Hg
            w = np.linalg.eigvalsh(H.tocsr()[sec][:, sec].toarray())
            resid, rank = np.linalg.norm(Hg - E * gs), int(np.sum(w < E - 1e-9))
        return gs, resid, rank, "phi from z.npy (non-impurity solver)"
    gs, e2 = impurity_gs(V, lam_c, ops)
    return gs, 0.0, 0, "ground state of H_imp (impurity solver)"


def ph_phase(n, m):
    """Phase (-1)^(N(N-1)/2) of the particle-hole map (N = number of occupied modes of |n>): reversal of the order of the N creation
    operators. With it the gGA relations hold in their plain form, R S = Tr[Phi^T c^dag Phi f], with no fermion-parity factor;
    without it (U_PH as defined above) the parity factor has to be inserted and the covalent and ionic amplitudes of Psi come out
    with opposite relative sign (kinetic energy +1.41 instead of -1.41 at U/t=4)."""
    N = bin(n).count("1")
    return -1.0 if (N * (N - 1) // 2) % 2 else 1.0


def ph_matrix(Neff, phase=None):
    """U_PH on the bath/qp Fock space of 2*Neff modes: columns U[:, n] = U_PH|n>, |n> = f^dag_j1 ... f^dag_jk |0> (j1<...<jk),
    U_PH|n> = f_j1 ... f_jk |full>, |full> = f^dag_0 ... f^dag_{m-1}|0> (index 4^Neff - 1)."""
    m = 2 * Neff
    c = gg.build_c_ops(m)
    dim = 2 ** m
    U = np.zeros((dim, dim))
    for n in range(dim):
        js = [j for j in range(m) if (n >> (m - 1 - j)) & 1]
        v = np.zeros(dim); v[dim - 1] = 1.0
        for j in reversed(js):            # f_jk acts first
            v = c[j] @ v
        U[:, n] = v * (phase or ph_phase)(n, m)
    return U


# ------------------------------------------------------------------ qp Slater determinant as a Fock vector
def psi0_matrix(Wocc, Neff):
    """M[n0, n1]: Slater determinant of the Neff occupied orbitals Wocc (2Neff x Neff, columns) for each spin, in the site-major qp Fock space."""
    m = 4 * Neff
    c = gg.build_c_ops(m)
    mode = lambda i, a, s: i * 2 * Neff + 2 * a + s
    v = np.zeros(2 ** m); v[0] = 1.0
    for s in range(2):
        for k in range(Neff):
            # A^dag_{k s} = sum_{i,a} Wocc[i*Neff+a, k] f^dag_{i a s}
            Ad = sum(Wocc[i * Neff + a, k] * c[mode(i, a, s)].T for i in range(2) for a in range(Neff))
            v = Ad @ v
    return v.reshape(4 ** Neff, 4 ** Neff)


INV_TOL = float(os.environ.get("INV_TOL", "1e-12"))


def inv_sqrt_psd(P, Neff, tol=None):
    tol = INV_TOL if tol is None else tol
    """P^(-1/2) on the support of P, block by block in (N_up, N_dn) of the qp Fock basis (P0 is block diagonal; diagonalizing
    the blocks separately keeps the 1e-16 noise between blocks from being amplified by the small eigenvalues of a near-pure ghost)."""
    m = 2 * Neff
    bits = lambda n, par: sum((n >> (m - 1 - j)) & 1 for j in range(par, m, 2))
    key = [(bits(n, 0), bits(n, 1)) for n in range(4 ** Neff)]
    out = np.zeros_like(P); nnull = 0; null_vecs = []
    for k in sorted(set(key)):
        ix = np.array([n for n in range(4 ** Neff) if key[n] == k])
        d, X = np.linalg.eigh(P[np.ix_(ix, ix)])
        inv = np.where(d > tol, 1.0 / np.sqrt(np.clip(d, tol, None)), 0.0)
        out[np.ix_(ix, ix)] = (X * inv) @ X.T
        for j in np.flatnonzero(d <= tol):
            v = np.zeros(4 ** Neff); v[ix] = X[:, j]; null_vecs.append(v)
        nnull += int(np.sum(d <= tol))
    return out, nnull, (np.array(null_vecs).T if null_vecs else np.zeros((4 ** Neff, 0)))


# ------------------------------------------------------------------ main builder
def build_psi(d, U, dV, Ng=None, verbose=True, phase=None):
    """Psi of the gGA point in directory d. Needs z.npy (non-impurity solver; then no R / lambda file is required, Psi0 is the Slater
    determinant of the column space of X) or the R / lambda files (impurity solver). Ng gives Neff = 1 + Ng when there is no R file."""
    import glob
    has_RL = bool(glob.glob(os.path.join(d, "*_R.dat")))
    R = lam = None
    if has_RL:
        R, lam = load_R_lambda(d); Neff = len(R[0])
    else:
        Neff = Ng + 1
    ops = [gg.build_impurity_ops(Neff, U, U / 2 + dV), gg.build_impurity_ops(Neff, U, U / 2 - dV)]
    sec = ops[0]["sector"]; m = len(sec)
    zf = os.path.join(d, "z.npy")
    z = np.load(zf) if os.path.exists(zf) else None
    use_z = z is not None and len(z) == 2 * m + 2 * Neff * Neff
    V = lamc = None
    if use_z:
        Wocc, _ = np.linalg.qr(z[2 * m:].reshape(2 * Neff, Neff))
        rho = Wocc @ Wocc.T
        Hqp = None
    else:
        Hqp, rho = gg.qp_state(R, lam, t)
        w, W = np.linalg.eigh(Hqp); Wocc = W[:, :Neff]
    blocks = [rho[:Neff, :Neff], rho[Neff:, Neff:]]
    if has_RL:
        V = [gg.hybridization_V(rho, R, t, SHIFT, i) for i in range(2)]
        lamc = [gg.embedding_lambda_c(blocks[i], lam[i], V[i], R[i], SHIFT) for i in range(2)]
    UPH = ph_matrix(Neff, phase)
    M = psi0_matrix(Wocc, Neff)
    P0 = [M @ M.T, M.T @ M]
    c_phys = gg.build_c_ops(2)                  # [up, dn] of one physical orbital
    f_qp = gg.build_c_ops(2 * Neff)

    lams, info = [], []
    for i in range(2):
        gs, resid, rank, label = embedding_vector(d, i, ops[i], Neff, None if V is None else V[i], None if lamc is None else lamc[i])
        docc = float(gs @ (ops[i]["n_pu"] @ ops[i]["n_pd"] @ gs)); nph = float(gs @ ((ops[i]["n_pu"] + ops[i]["n_pd"]) @ gs))
        Phi = gs.reshape(4, 4 ** Neff)                       # [Gamma, nbar]
        Phit = Phi @ UPH                                     # Phi = Phit U_PH^T  ->  Phit = Phi U_PH
        # recompute R from Phit:  T_a = Tr[Phit^T c_up^dag Phit f_{a up}] = D_a
        X = Phit.T @ (c_phys[0].T @ Phit)
        T = np.array([np.trace(X @ f_qp[2 * a].toarray()) for a in range(Neff)])
        dd, W, s_ = gg._sqrt_eig(blocks[i], SHIFT)
        Rchk = W @ ((W.T @ T) / s_)
        # T is quadratic in Phit, so the overall sign of the embedding vector is irrelevant; what is NOT a free choice is the relative
        # sign of the odd-Gamma and even-Gamma rows (it flips R_i and, in Psi, the sign of the covalent against the ionic amplitude):
        # when the solution's R is available it is fixed by requiring R(from Phit) = +R of the solution (whose sign enters H_qp through
        # the hopping -t R_0 R_1); with the phase of ph_phase it never needed flipping (checked on all the points of the doc)
        flipped = False
        if R is not None and (np.sign(np.dot(Rchk, R[i])) or 1.0) < 0:
            Phit[[1, 2], :] *= -1.0; Rchk = -Rchk; flipped = True
        Pinv, nnull, null = inv_sqrt_psd(P0[i], Neff)
        wnull = np.linalg.norm(Phit @ null) ** 2 if null.shape[1] else 0.0
        lams.append(Phit @ Pinv)
        info.append(dict(Rchk=Rchk, R=None if R is None else R[i], nnull=int(nnull), wnull=wnull, flipped=flipped,
                         docc=docc, nphys=nph, resid=resid))
        if verbose:
            print(f"   site {i}: R(file)={None if R is None else np.round(R[i], 5)}  R(from Phit)={np.round(Rchk, 5)}  "
                  f"null(P0)={int(nnull)}  |Phit|^2 on null={wnull:.2e}  odd rows flipped: {flipped}"
                  f"\n           embedding state: {label}; residual={resid}, {rank} eigenstates below it", flush=True)

    # parity of lambda: parity(Gamma) - parity(n) should be constant (even) for the 2-site JW composition used below
    par = lambda x: bin(x).count("1") % 2
    bad = sum(1 for L in lams for G in range(4) for n in range(4 ** Neff) if abs(L[G, n]) > 1e-10 and par(G) != par(n))
    if verbose and bad:
        print(f"   WARNING: {bad} parity-odd entries in lambda: the two-site composition needs fermionic signs")

    Psi = (lams[0] @ M @ lams[1].T).reshape(16)
    Rs = [info[0]["Rchk"], info[1]["Rchk"]]
    kin_gga = -4.0 * t * Rs[0] @ rho[:Neff, Neff:] @ Rs[1]       # E_kin = -4 t R_0 rho_01 R_1 of the gGA functional
    return dict(Psi=Psi, Hqp=Hqp, R=R, lam=lam, info=info, Neff=Neff, M=M, lams=lams, kin_gga=kin_gga, rho=rho)


# ------------------------------------------------------------------ analysis in the physical space
def physical(U, dV):
    # gGA lowers site 0 (mu_0 = U/2 + dV, H_loc = U n n - mu_0 n); exact_dimer_deltaV raises it by Delta V/2: Delta V_ED = -2 dV
    H, c, n0, n1 = ed.build_hamiltonian(t, U, -2 * dV)
    N = np.real(np.diag(n0 + n1))
    return H, c, n0, n1, N


def cov_ion_basis():
    """|cov> = (c0u^dag c1d^dag - c0d^dag c1u^dag)/sqrt2, |ion0> = c0u^dag c0d^dag, |ion1> = c1u^dag c1d^dag, as vectors (index bits: modes 0u,0d,1u,1d)."""
    idx = lambda *modes: sum(1 << (3 - m) for m in modes)
    cov = np.zeros(16); cov[idx(0, 3)] = 1 / np.sqrt(2); cov[idx(1, 2)] = -1 / np.sqrt(2)
    i0 = np.zeros(16); i0[idx(0, 1)] = 1.0
    i1 = np.zeros(16); i1[idx(2, 3)] = 1.0
    return cov, i0, i1


def analyse(Psi, U, dV):
    H, c, n0, n1, N = physical(U, dV)
    nrm = Psi @ Psi
    p = Psi / np.sqrt(nrm)
    cov, i0, i1 = cov_ion_basis()
    n0u = c[0].T @ c[0]; n0d = c[1].T @ c[1]; n1u = c[2].T @ c[2]; n1d = c[3].T @ c[3]
    pion = p @ (n0u @ n0d + n1u @ n1d) @ p
    # exact ground state (N=2 sector) and exact hole spectrum (N=1 sector)
    sec2 = np.flatnonzero(np.abs(N - 2) < 1e-9); sec1 = np.flatnonzero(np.abs(N - 1) < 1e-9)
    w2, v2 = np.linalg.eigh(H[np.ix_(sec2, sec2)])
    g = np.zeros(16); g[sec2] = v2[:, 0]
    w1, v1 = np.linalg.eigh(H[np.ix_(sec1, sec1)])

    def hole(state):
        out = []
        for m in range(len(w1)):
            vec = np.zeros(16); vec[sec1] = v1[:, m]
            out.append(sum((vec @ (c[k] @ state)) ** 2 for k in range(4)))
        return np.array(out)

    hop = -(c[0].T @ c[2] + c[2].T @ c[0] + c[1].T @ c[3] + c[3].T @ c[1])
    pion_ed = g @ (n0u @ n0d + n1u @ n1d) @ g
    sg = np.sign(cov @ g) or 1.0
    return dict(amp_cov_ed=sg * (cov @ g), amp_ion_ed=sg * (i0 + i1) @ g / np.sqrt(2), amp_ionasym_ed=sg * (i0 - i1) @ g / np.sqrt(2), norm=nrm, p=p, E=p @ H @ p, E_ed=w2[0], pion_ed=pion_ed, kin=p @ hop @ p, kin_ed=g @ hop @ g, ovl=abs(p @ g), pion=pion,
                amp_cov=cov @ p, amp_ion=(i0 + i1) @ p / np.sqrt(2), amp_ionasym=(i0 - i1) @ p / np.sqrt(2),
                nsec=np.linalg.norm(p[sec2]), poles=w2[0] - w1, w_psi=hole(p), w_ed=hole(g))


def _branches(path):
    br = {}
    for line in open(path):
        if not line.startswith("#") and line.strip():
            f = line.split(); br[round(float(f[0]), 1)] = f[2]
    return br


def solution_dir(which, ng, U):
    """(directory of the point, dV). dV0 / dV2 use the lowest-energy branch of the scans (lowest_energy.txt) where there are two."""
    if os.path.isdir(which):                                  # any directory with U_<U>/ subdirectories
        return which.rstrip("/") + f"/U_{U:.1f}", float(os.environ.get("DV", "0"))
    dV = 0.0 if which == "dV0" else 2.0
    if which == "dV2n":                                       # non-impurity scan at dV=2 (z.npy), lowest-energy branch
        base = f"../scan_ng{ng}_not_imp_dV2"
        return f"{base}/{_branches(base + '/lowest_energy.txt')[round(U, 1)]}/U_{U:.1f}", 2.0
    if ng == 0:
        base = f"../scan_ng0/dV_{dV:.1f}"
        return f"{base}/{_branches(base + '/lowest_energy.txt')[round(U, 1)]}/U_{U:.1f}", dV
    if which == "dV0":
        if ng == 2:
            return f"../scan_ng2_not_imp/{_branches('../scan_ng2_not_imp/lowest_energy.txt')[round(U, 1)]}/U_{U:.1f}", 0.0
        return f"../scan_ng4_not_imp/U_{U:.1f}", 0.0
    return f"../scan_ng{ng}_dV2_confirmed/U_{U:.1f}", 2.0


def gga_energy(d):
    import re
    m = re.search(r"E_var/t=\s*([-\d.eE+]+)", open(os.path.join(d, "output.out")).read())
    return float(m.group(1)) if m else float("nan")


def report(which, ng, U, verbose=True):
    d, dV = solution_dir(which, ng, U)
    r = build_psi(d, U, dV, Ng=ng, verbose=verbose)
    a = analyse(r["Psi"], U, dV)
    a.update(U=U, dV=dV, Ng=ng, dir=d, E_gga=gga_energy(d), kin_gga=r["kin_gga"],
             pion_frag=r["info"][0]["docc"] + r["info"][1]["docc"], Rchk=[i["Rchk"] for i in r["info"]],
             wnull=max(i["wnull"] for i in r["info"]))
    return a


if __name__ == "__main__":
    which, ng = sys.argv[1], int(sys.argv[2])
    Us = [float(x) for x in sys.argv[3:]] or [1.0, 2.0, 4.0, 6.0]
    for U in Us:
        print(f"=== {which} Ng={ng} U/t={U}", flush=True)
        a = report(which, ng, U)
        print(f"   ({a['dir']})  <Psi|Psi>={a['norm']:.6f}  N=2-sector weight={a['nsec']:.6f}")
        print(f"   amplitudes (normalized Psi): cov={a['amp_cov']:+.5f}  ion(sym)={a['amp_ion']:+.5f}  ion(asym)={a['amp_ionasym']:+.5f}   "
              f"P_ion(Psi)={a['pion']:.5f}  P_ion(ED)={a['pion_ed']:.5f}  P_ion(fragments)={a['pion_frag']:.5f}")
        print(f"   E_Psi={a['E']:.6f}   E_FCI={a['E_ed']:.6f}   E_gGA={a['E_gga']:.6f}   overlap with the exact ground state={a['ovl']:.6f}")
        print(f"   kinetic energy: Psi={a['kin']:+.5f}  exact={a['kin_ed']:+.5f}  gGA functional (-4t R0 rho01 R1)={a['kin_gga']:+.5f}")
        print("   [ED cation states, not a gGA spectrum: Psi is projected on the exact N=1 eigenstates of H]")
        print("      exact hole poles           : " + "  ".join(f"{x:8.4f}" for x in a["poles"]))
        print("      |<cation|c|Psi>|^2          : " + "  ".join(f"{x:8.4f}" for x in a["w_psi"]))
        print("      |<cation|c|exact GS>|^2     : " + "  ".join(f"{x:8.4f}" for x in a["w_ed"]), flush=True)
