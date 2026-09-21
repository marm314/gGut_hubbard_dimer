"""Ghost Gutzwiller Approximation (gGA) for the half-filled 2-site Hubbard dimer.

Each site is its own impurity/fragment (the two fragments are NOT assumed
equal). Ng is the number of *extra* ghost bath orbitals beyond the one bath
orbital standard Gutzwiller already has; Neff = 1 + Ng is the number of
quasiparticle/bath orbitals per fragment per spin.

The self-consistency cycle deliberately mirrors, step by step, the reference
C++ code (GhostGutzwiller::Run with no_fit_scf, the 'lin' updater, and the
msite_ggm matrix basis); only the impurity problem is solved by our own
solver. One iteration is:

  1. lambda_i = -lambda^c_i + G(Delta_i, V_i R_i)          (previous V, lambda^c;
     iteration 0 instead starts from the input lambda)
  2. H_qp = [[-lambda_0, -t R_0^T R_1], [-t R_1^T R_0, -lambda_1]]; the qp
     ground state fills the lowest Neff of the 2*Neff levels (the C++
     RSImpMolecule rule nup_eff = mol_nups + (Neff_tot - Nphys_tot)/2);
     Delta_i is its fragment-diagonal block, rho_ij its off-diagonal blocks.
  3. V_i = S(Delta_i)^-1 X_i,   X_i = -t rho_ij R_j          (Eq. 7)
  4. lambda^c_i = -lambda_i + G(Delta_i, V_i R_i)           (Eq. 8)
  5. impurity ground state -> Delta^new_i = 1 - <d^dag d>,
     R^new_i = S(Delta^new_i)^-1 <d^dag c>                  (Eq. 10)
  6. mixing: plain substitution for the first `eq_time` iterations, then
     linear mixing with weight `mix`; Delta is then clipped to [0, 1].

(The impurity ground state is searched in the half-filled, spin-balanced
sector N_up = N_dn = (1+Neff)/2 only, as in the C++ CMZEDsolver, and V and
lambda^c are passed to it through a 12-digit / 1e-10-threshold quantization,
as in the C++ FCIDUMP hand-over.)

with S(D) = sqrt(D(1-D) + shift)  (the shift is added to the *eigenvalues*
of D(1-D), as in the C++ GetMatSqrt, and Delta itself is never clipped
before that) and G the derivative of tr[M S(Delta)] projected on symmetric
matrices (orthonormal generalized Gell-Mann basis), evaluated analytically by
solving the Sylvester equation in the eigenbasis of Delta.

Dimension bookkeeping: with Nphys=1 (one physical orbital per fragment, spins
handled symmetrically), the embedding Hamiltonian has Nphys+Neff = 2+Ng
orbitals per spin, so the impurity Fock space has dimension
2**(2*(2+Ng)) = 4**(2+Ng).
"""

import os

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh


# ---------------------------------------------------------------------------
# Jordan-Wigner fermion operators (generic n_modes)
# ---------------------------------------------------------------------------

# The operators are stored sparse: the Fock space has dimension 4**(2+Ng)
# (4096 for Ng=4), where the dense matrices would need ~10 GB in total.
I2 = sp.identity(2, format="csr")
Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
C_LOCAL = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


def build_c_ops(n_modes):
    ops = []
    for i in range(n_modes):
        mats = [Z if j < i else (C_LOCAL if j == i else I2) for j in range(n_modes)]
        ops.append(_kron_list(mats))
    return ops


# ---------------------------------------------------------------------------
# Quasiparticle side (mirrors the C++ reference code)
# ---------------------------------------------------------------------------

def _sqrt_eig(Delta, shift):
    """Eigendecomposition of Delta and the eigenvalues s_i = sqrt(d_i(1-d_i) + shift)
    of S(Delta) = sqrt(Delta(1-Delta) + shift), as in the C++ GetMatSqrt."""
    d, W = np.linalg.eigh(Delta)
    return d, W, np.sqrt(d * (1.0 - d) + shift)


def apply_S_inv(Delta, B, shift):
    """S(Delta)^-1 @ B, with B a vector or a matrix."""
    d, W, s = _sqrt_eig(Delta, shift)
    Bt = W.T @ B
    return W @ (Bt / (s[:, None] if Bt.ndim == 2 else s))


def qp_state(R, lam, t):
    """Quasiparticle Hamiltonian and its zero-temperature ground-state 1-RDM
    (per spin) for the two fragments, R = [R0, R1], lam = [lam0, lam1].

    The occupation is the C++ RSImpMolecule one: a fixed LEVEL COUNT, the
    lowest nup_eff = mol_nups + (Neff_tot - Nphys_tot)/2 = Neff eigenvectors
    of the 2*Neff of H_qp are filled (per spin), whatever the signs of their
    eigenvalues. (The base-class rule "e < 0" is NOT what molecules use.)"""
    Neff = len(R[0])
    Hqp = np.zeros((2 * Neff, 2 * Neff))
    Hqp[:Neff, :Neff] = -lam[0]
    Hqp[Neff:, Neff:] = -lam[1]
    Hqp[:Neff, Neff:] = -t * np.outer(R[0], R[1])
    Hqp[Neff:, :Neff] = -t * np.outer(R[1], R[0])
    evals, evecs = np.linalg.eigh(Hqp)
    occ = np.zeros(2 * Neff)
    occ[:Neff] = 1.0   # eigh returns ascending eigenvalues
    rho = (evecs * occ) @ evecs.T
    return Hqp, rho


def hybridization_V(rho, R, t, shift, i):
    """Eq. 7: S(Delta_i) V_i = X_i, with X_i = -t rho_ij R_j (the other fragment's R)."""
    Neff = len(R[0])
    j = 1 - i
    rho_ii = rho[i * Neff:(i + 1) * Neff, i * Neff:(i + 1) * Neff]
    rho_ij = rho[i * Neff:(i + 1) * Neff, j * Neff:(j + 1) * Neff]
    return apply_S_inv(rho_ii, -t * (rho_ij @ R[j]), shift)


def grad_S(Delta, M, shift):
    """Symmetric derivative G of tr[M S(Delta)] with respect to Delta:

        G = sum_k B_k * 2 tr(M dS[B_k]),   B_k an orthonormal basis of the
                                            real symmetric matrices,

    the C++ eigspaceOneRDMDerivator sum. In the eigenbasis of Delta (M~ = W^T M W)
    the Sylvester equation is diagonal,
        dS[B]_ij = B_ij (1 - d_i - d_j) / (s_i + s_j),
    which gives G = W [(M~ + M~^T) o w] W^T with w_ij = (1-d_i-d_j)/(s_i+s_j).
    Exact (no finite differences), and finite for eigenvalues of Delta at 0 or 1
    thanks to the shift."""
    d, W, s = _sqrt_eig(Delta, shift)
    w = (1.0 - d[:, None] - d[None, :]) / (s[:, None] + s[None, :])
    Mt = W.T @ M @ W
    return W @ ((Mt + Mt.T) * w) @ W.T


def embedding_lambda_c(Delta, lam, V, R, shift):
    """Eq. 8: lambda^c = -lambda + G(Delta, V R) (V column, R row)."""
    return -lam + grad_S(Delta, np.outer(V, R), shift)


# ---------------------------------------------------------------------------
# Impurity (embedding) side
# ---------------------------------------------------------------------------

def build_impurity_ops(Neff, U, mu):
    """Precompute all fixed operators/matrices once per (Neff, U) -- nothing
    here depends on the self-consistency loop's V, lambda^c, so it must not
    be rebuilt every iteration (that was the earlier O(iterations * dim^3)
    bottleneck)."""
    n_modes = 2 * (1 + Neff)
    c = build_c_ops(n_modes)
    c_pu, c_pd = c[0], c[1]
    bath_up = [c[2 + 2 * k] for k in range(Neff)]
    bath_dn = [c[3 + 2 * k] for k in range(Neff)]

    n_pu = c_pu.conj().T @ c_pu
    n_pd = c_pd.conj().T @ c_pd
    H_loc = U * (n_pu @ n_pd) - mu * (n_pu + n_pd)

    hyb_up = [bath_up[a].conj().T @ c_pu + c_pu.conj().T @ bath_up[a] for a in range(Neff)]
    hyb_dn = [bath_dn[a].conj().T @ c_pd + c_pd.conj().T @ bath_dn[a] for a in range(Neff)]

    # Precompute bath-bath bilinears once -- these get reused, weighted by
    # lambda^c, every outer iteration; building them on the fly per
    # iteration was the O(iterations * Neff^2 * dim^3) bottleneck.
    lam_up = [[bath_up[a].conj().T @ bath_up[b] for b in range(Neff)] for a in range(Neff)]
    lam_dn = [[bath_dn[a].conj().T @ bath_dn[b] for b in range(Neff)] for a in range(Neff)]

    # Particle-number sector searched by the ground-state solve. Like the C++
    # CMZEDsolver (even number of orbitals per spin), only the half-filled,
    # spin-balanced sector N_up = N_dn = (1+Neff)/2 is diagonalized, not the
    # whole Fock space: the global minimum of H_imp can lie in another sector
    # (e.g. a triplet) far from self-consistency, which the reference never
    # visits. Mode j is occupied in basis state idx iff bit (n_modes-1-j) of
    # idx is set (Jordan-Wigner kron order, index 1 = occupied).
    n_orb = 1 + Neff
    idx = np.arange(2 ** n_modes)
    occ = lambda j: (idx >> (n_modes - 1 - j)) & 1
    n_up = sum(occ(j) for j in range(0, n_modes, 2))
    n_dn = sum(occ(j) for j in range(1, n_modes, 2))
    sector = np.flatnonzero((n_up == n_orb // 2) & (n_dn == n_orb // 2))

    return {
        "c_pu": c_pu, "c_pd": c_pd, "bath_up": bath_up, "bath_dn": bath_dn,
        "n_pu": n_pu, "n_pd": n_pd, "H_loc": H_loc,
        "hyb_up": hyb_up, "hyb_dn": hyb_dn,
        "lam_up": lam_up, "lam_dn": lam_dn, "Neff": Neff, "sector": sector,
    }


def fcidump_round(x):
    """Emulate the C++ hand-over of the impurity Hamiltonian to the ED solver
    through a text FCIDUMP file (WriteImpFCIDUMP): every element is written
    with std::scientific at precision 12 (i.e. '%.12e'), and elements with
    |x| < 1e-10 are not written at all (read back as zero). This quantization
    is part of what the reference does before its solver runs, and it matters
    for cold starts: the R update S^-1 D divides by sqrt(n(1-n)+shift), so
    without it noise far below 1e-10 is amplified into O(1) changes of R and
    the iteration flips between branches; with it the trajectory is
    reproducible (see doc/code.tex, Section on validation)."""
    x = np.asarray(x, dtype=float)
    y = np.array([float("%.12e" % v) for v in x.ravel()]).reshape(x.shape)
    return np.where(np.abs(x) < 1e-10, 0.0, y)


def impurity_solve(V, lam_c, ops):
    Neff = ops["Neff"]
    c_pu, c_pd = ops["c_pu"], ops["c_pd"]
    bath_up, bath_dn = ops["bath_up"], ops["bath_dn"]
    H_loc = ops["H_loc"]

    H_imp = H_loc
    for a in range(Neff):
        H_imp = H_imp + V[a] * (ops["hyb_up"][a] + ops["hyb_dn"][a])

    # Eq. 30 (NotesOngGut.pdf): the bath term is -sum_ab lambda^c_ab d_b d^dag_a.
    # Using {d_a,d^dag_b}=delta_ab, d_b d^dag_a = delta_ab - d^dag_a d_b, so
    # normal-ordered this is +sum_ab lambda^c_ab d^dag_a d_b (plus an
    # irrelevant additive constant) -- a PLUS sign, not minus.
    for a in range(Neff):
        for b in range(Neff):
            if lam_c[a, b] == 0.0:
                continue
            H_imp = H_imp + lam_c[a, b] * (ops["lam_up"][a][b] + ops["lam_dn"][a][b])

    # Ground state in the N_up = N_dn = (1+Neff)/2 sector only (see
    # build_impurity_ops), embedded back into the full Fock space.
    sec = ops["sector"]
    evals, evecs = eigh(H_imp.tocsr()[sec][:, sec].toarray())
    gs = np.zeros(H_imp.shape[0])
    gs[sec] = evecs[:, 0]

    # Expectation values via matrix-vector products only (O(Neff*dim^2)),
    # never forming the O(dim x dim) operator products A^dag B explicitly.
    bu_gs = [bath_up[a] @ gs for a in range(Neff)]
    bd_gs = [bath_dn[a] @ gs for a in range(Neff)]
    cpu_gs = c_pu @ gs
    cpd_gs = c_pd @ gs

    Delta_bb = np.zeros((Neff, Neff))
    for a in range(Neff):
        for b in range(Neff):
            val_u = np.vdot(bu_gs[a], bu_gs[b])
            val_d = np.vdot(bd_gs[a], bd_gs[b])
            Delta_bb[a, b] = 0.5 * (val_u + val_d).real

    D = np.zeros(Neff)
    for a in range(Neff):
        val_u = np.vdot(bu_gs[a], cpu_gs)
        val_d = np.vdot(bd_gs[a], cpd_gs)
        D[a] = 0.5 * (val_u + val_d).real

    E_loc = (gs.conj() @ (H_loc @ gs)).real
    docc = (gs.conj() @ (ops["n_pu"] @ ops["n_pd"] @ gs)).real
    n_phys = (gs.conj() @ ((ops["n_pu"] + ops["n_pd"]) @ gs)).real

    return Delta_bb, D, E_loc, docc, n_phys


# ---------------------------------------------------------------------------
# Self-consistency loop
# ---------------------------------------------------------------------------

def run_gga(Ng, U, t=1.0, max_iter=200, tol_E=1e-6, tol_mat=1e-6, mix=0.65, eq_time=10,
            sqmat_shift=1e-9, round_fcidump=True, verbose=False,
            Rg0=None, Rg1=None, lamg0=None, lamg1=None):
    """Self-consistency loop for the 2-site dimer, treating the two fragments as
    INDEPENDENT (no site-exchange symmetry assumed). See the module docstring
    for the cycle, which follows the reference C++ code step by step.

    Convergence (as in the C++ GhostGutTerminationTracker): |dE| < tol_E, or
    both ddelta < tol_mat and dR < tol_mat, where ddelta and dR are the
    summed Frobenius norms (over fragments) of the change of the next
    Delta / R with respect to the previous / current ones."""
    import time

    Neff = 1 + Ng
    mu = U / 2.0
    fock_dim = 2 ** (2 * (1 + Neff))

    if verbose:
        print(f"[setup] Ng={Ng}  Neff={Neff}  impurity orbitals/spin={1+Neff}  "
              f"Fock dim={fock_dim}  (sparse operators)", flush=True)
        t_build0 = time.time()

    ops = build_impurity_ops(Neff, U, mu)

    if verbose:
        print(f"[setup] built impurity operators in {time.time()-t_build0:.2f}s; "
              f"ground state searched in the N_up=N_dn sector of dim {len(ops['sector'])}", flush=True)

    if Rg0 is not None:
        R0 = np.array(Rg0, dtype=float)
    else:
        R0 = np.zeros(Neff)
        R0[0] = 0.9
        if Neff > 1:
            # break ghost-exchange symmetry: distinct values, not a repeated
            # constant, else the iteration sits on the degenerate manifold
            # shared by the trivial "sleeping ghost" fixed points
            R0[1:] = 0.05 * (1.0 + 0.3 * np.arange(Neff - 1))
    R1 = np.array(Rg1, dtype=float) if Rg1 is not None else R0.copy()
    R = [R0, R1]

    lam0 = np.array(lamg0, dtype=float) if lamg0 is not None else np.zeros((Neff, Neff))
    lam1 = np.array(lamg1, dtype=float) if lamg1 is not None else lam0.copy()
    lam = [lam0, lam1]

    lamc = [np.zeros((Neff, Neff)), np.zeros((Neff, Neff))]
    V = [np.zeros(Neff), np.zeros(Neff)]

    def blocks(rho):
        return [rho[:Neff, :Neff], rho[Neff:, Neff:]]

    curr_E = 100.0   # same starting value as the C++ tracker
    n_updates = 0    # calls of the linear mixer
    res = None
    converged = False
    for it in range(max_iter):
        t_iter0 = time.time()

        if it == 0:
            # start from the input lambda: Delta from the qp ground state
            Hqp, rho = qp_state(R, lam, t)
            Delta = blocks(rho)
            prev_Delta = [d.copy() for d in Delta]
        else:
            # lambda from the previous lambda^c and V, at the (mixed) Delta and R
            lam = [embedding_lambda_c(Delta[i], lamc[i], V[i], R[i], sqmat_shift) for i in range(2)]
            Hqp, rho = qp_state(R, lam, t)
            Delta = blocks(rho)

        V = [hybridization_V(rho, R, t, sqmat_shift, i) for i in range(2)]
        lamc = [embedding_lambda_c(Delta[i], lam[i], V[i], R[i], sqmat_shift) for i in range(2)]
        t_qp = time.time()

        # The C++ hands V and lambda^c to its solver through a text FCIDUMP
        # (12 digits, elements < 1e-10 dropped); emulate that (see fcidump_round).
        rnd = fcidump_round if round_fcidump else (lambda x: x)
        imp = [impurity_solve(rnd(V[i]), rnd(lamc[i]), ops) for i in range(2)]
        t_solve = time.time()

        Delta_new = [np.eye(Neff) - imp[i][0] for i in range(2)]
        R_new = [apply_S_inv(Delta_new[i], imp[i][1], sqmat_shift) for i in range(2)]

        E_loc = imp[0][2] + imp[1][2]
        Lblk = np.zeros_like(Hqp)
        Lblk[:Neff, :Neff] = lam[0]
        Lblk[Neff:, Neff:] = lam[1]
        E_qp = 2.0 * np.trace(rho @ (Hqp + Lblk))   # 2 = spin
        E_var = E_qp + E_loc
        docc = 0.5 * (imp[0][3] + imp[1][3])
        n_phys = 0.5 * (imp[0][4] + imp[1][4])

        # Linear mixer (C++ LinearMixer): substitution during the first
        # eq_time calls, then mixing with weight `mix`; then clip Delta to [0,1].
        n_updates += 1
        a = 1.0 if n_updates <= eq_time else mix
        next_Delta = [Delta[i] + a * (Delta_new[i] - Delta[i]) for i in range(2)]
        next_R = [R[i] + a * (R_new[i] - R[i]) for i in range(2)]
        for i in range(2):
            d, W = np.linalg.eigh(next_Delta[i])
            next_Delta[i] = (W * np.clip(d, 0.0, 1.0)) @ W.T

        ddelta = sum(np.linalg.norm(prev_Delta[i] - next_Delta[i]) for i in range(2))
        dR = sum(np.linalg.norm(R[i] - next_R[i]) for i in range(2))
        dE = E_var - curr_E

        if verbose:
            print(f"  it={it:3d}  Z0={float(R[0] @ R[0]):.6f}  Z1={float(R[1] @ R[1]):.6f}  "
                  f"E_var={E_var:.6f}  dE={dE:.3e}  docc={docc:.6f}  n_phys={n_phys:.6f}  "
                  f"|dDelta|={ddelta:.3e}  |dR|={dR:.3e}  "
                  f"[qp {t_qp-t_iter0:.2f}s | ED {t_solve-t_qp:.2f}s]", flush=True)

        # Everything reported below belongs to this iteration's (R, lam, V, lamc);
        # R is the updated one, as the C++ code saves it (final_R).
        res = dict(Delta=[d.copy() for d in Delta], V=[v.copy() for v in V],
                   lam=[l.copy() for l in lam], lamc=[l.copy() for l in lamc],
                   E_var=E_var, docc=docc, n_phys=n_phys)

        prev_Delta = [d.copy() for d in next_Delta]
        R = next_R
        Delta = next_Delta
        curr_E = E_var

        if abs(dE) < tol_E or (ddelta < tol_mat and dR < tol_mat):
            converged = True
            break

    Z0 = float(np.dot(R[0], R[0]))
    Z1 = float(np.dot(R[1], R[1]))

    return {
        "Ng": Ng, "U": U, "R0": R[0], "R1": R[1], "Z0": Z0, "Z1": Z1, "E_var": res["E_var"],
        "docc": res["docc"], "n_phys": res["n_phys"], "iters": it, "converged": converged,
        "lam0": res["lam"][0], "lam1": res["lam"][1],
        "lamc0": res["lamc"][0], "lamc1": res["lamc"][1], "V0": res["V"][0], "V1": res["V"][1],
    }


def compute_spectral_function(R0, R1, lam0, lam1, t, omega_max, eta, domega):
    """Quasiparticle Green's function G(omega) = [(omega + i*eta) I - H_qp]^-1,
    on a uniform grid from -omega_max to +omega_max in steps of domega,
    transformed to the physical (site) basis via the block-diagonal
    renormalization matrix Rmat = diag(R0, R1) (shape (2*Neff, 2)):

        G_phys(omega) = Rmat^T . G(omega) . Rmat        (2x2, site0/site1)

    The physical spectral function is A(omega) = -Im[Tr G_phys(omega)] / pi,
    doubled to account for both (degenerate) spin channels, since H_qp above
    is the same one-body problem for either spin."""
    Neff = len(R0)
    dim = 2 * Neff
    Hqp = np.zeros((dim, dim))
    Hqp[:Neff, :Neff] = -lam0
    Hqp[Neff:, Neff:] = -lam1
    Hqp[:Neff, Neff:] = -t * np.outer(R0, R1)
    Hqp[Neff:, :Neff] = -t * np.outer(R1, R0)

    Rmat = np.zeros((dim, 2))
    Rmat[:Neff, 0] = R0
    Rmat[Neff:, 1] = R1

    ident = np.eye(dim)
    # np.arange with a half-step pad so the closed endpoint +omega_max is
    # included despite floating-point round-off (plain np.arange(-a, a,
    # step) can drop the last point).
    omega_grid = np.arange(-omega_max, omega_max + 0.5 * domega, domega)
    A = np.empty(len(omega_grid))
    for i, w in enumerate(omega_grid):
        G = np.linalg.inv((w + 1j * eta) * ident - Hqp)
        G_phys = Rmat.T @ G @ Rmat
        A[i] = -2.0 * np.trace(G_phys).imag / np.pi
    return omega_grid, A


def save_spectral_function(omega_grid, A, path="A_omega.txt"):
    np.savetxt(path, np.column_stack([omega_grid, A]), header="omega   A(omega)")
    print(f"[saved] spectral function -> {path}")


def save_results(res, prefix=None):
    """Write the converged R vectors and lambda/lambda^c matrices to files,
    one row-block per fragment (atom 0 then atom 1) -- the same layout
    read_R_guess/read_lambda_guess accept back in as a two-atom guess."""
    if prefix is None:
        prefix = f"gga_Ng{res['Ng']}_U{res['U']:g}"

    r_file = f"{prefix}_R.dat"
    lam_file = f"{prefix}_lambda.dat"
    lamc_file = f"{prefix}_lambda_c.dat"

    Neff = len(res["R0"])
    np.savetxt(r_file, np.vstack([res["R0"], res["R1"]]),
               header=f"Ng={res['Ng']} U={res['U']} Neff={Neff}  converged R, one row per atom (atom0, atom1)")
    np.savetxt(lam_file, np.vstack([res["lam0"], res["lam1"]]),
               header=f"Ng={res['Ng']} U={res['U']}  qp-side lambda, stacked Neff x Neff blocks per atom (atom0, atom1)")
    np.savetxt(lamc_file, np.vstack([res["lamc0"], res["lamc1"]]),
               header=f"Ng={res['Ng']} U={res['U']}  impurity-side lambda^c, stacked Neff x Neff blocks per atom (atom0, atom1)")

    print(f"[saved] R -> {r_file}")
    print(f"[saved] lambda -> {lam_file}")
    print(f"[saved] lambda_c -> {lamc_file}")


def read_R_guess(path):
    """Read R guess(es) from a text file. One row (Neff numbers) is used as
    a symmetric guess for both atoms; two rows give each atom its own R
    (as reference calculations show is generally necessary -- the two
    fragments are not required to stay equal)."""
    arr = np.atleast_2d(np.loadtxt(path, delimiter=None)).astype(float)
    if arr.shape[0] == 1:
        return arr[0], arr[0].copy()
    if arr.shape[0] == 2:
        return arr[0], arr[1]
    raise ValueError(f"{path}: expected 1 row (shared guess) or 2 rows (one per atom), got {arr.shape[0]}")


def read_lambda_guess(path):
    """Read lambda guess(es) from a text file. Neff rows (a single Neff x
    Neff matrix) are used as a symmetric guess for both atoms; 2*Neff rows
    (two stacked Neff x Neff blocks) give each atom its own lambda."""
    arr = np.atleast_2d(np.loadtxt(path, delimiter=None)).astype(float)
    Neff = arr.shape[1]
    if arr.shape[0] == Neff:
        return arr, arr.copy()
    if arr.shape[0] == 2 * Neff:
        return arr[:Neff], arr[Neff:]
    raise ValueError(f"{path}: expected {Neff} rows (shared guess) or {2*Neff} rows "
                      f"(stacked per-atom blocks), got {arr.shape[0]}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="gGA self-consistency for the half-filled Hubbard dimer")
    parser.add_argument("--U", type=float, default=None, help="Hubbard U/t (single run)")
    parser.add_argument("--Ng", type=int, default=None, help="number of ghost bath orbitals (must be even)")
    parser.add_argument("--t", type=float, default=1.0, help="hopping (energy unit)")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-iteration trace (single-run mode is verbose by default)")
    parser.add_argument("--R-file", type=str, default="R.in", metavar="PATH",
                         help="file with the initial R guess (1 or 2 rows of Neff numbers); read automatically if it exists (default: R.in)")
    parser.add_argument("--lam-file", type=str, default="L.in", metavar="PATH",
                         help="file with the initial lambda guess (Neff or 2*Neff rows of Neff numbers); read automatically if it exists (default: L.in)")
    parser.add_argument("--max-iter", type=int, default=200, help="maximum outer self-consistency iterations (default: 200)")
    parser.add_argument("--sqmat-shift", type=float, default=1e-9,
                         help="shift added to the eigenvalues of Delta(1-Delta) before the square root "
                              "(C++ gg_sqmat_shift, default: 1e-9)")
    parser.add_argument("--mix", type=float, default=0.65,
                         help="linear mixing weight after the equilibration iterations (C++ gg_lin_mix, default: 0.65)")
    parser.add_argument("--eq-time", type=int, default=10,
                         help="number of initial iterations with plain substitution, no mixing (C++ gg_eq_time, default: 10)")
    parser.add_argument("--tol-E", type=float, default=1e-6, help="absolute energy tolerance (C++ gg_abs_tol_E, default: 1e-6)")
    parser.add_argument("--tol-mat", type=float, default=1e-6,
                         help="absolute tolerance for Delta and R changes (C++ gg_abs_tol_mat, default: 1e-6)")
    parser.add_argument("--no-fcidump-rounding", action="store_true",
                         help="do not emulate the C++ FCIDUMP round trip of V and lambda^c "
                              "(12 significant digits, elements below 1e-10 dropped); with it OFF, "
                              "cold-start trajectories are no longer reproducible against the C++ code")
    parser.add_argument("--omega", type=float, default=None,
                         help="max frequency for the spectral function grid, which runs from -omega to +omega; "
                              "requires --eta and --domega, and writes A_omega.txt after convergence")
    parser.add_argument("--eta", type=float, default=None,
                         help="broadening eta in the Green's function denominator (omega + i*eta); requires --omega and --domega")
    parser.add_argument("--domega", type=float, default=None,
                         help="frequency step size for the [-omega, omega] grid; requires --omega and --eta")
    args = parser.parse_args()

    if not (args.omega is None) == (args.eta is None) == (args.domega is None):
        parser.error("--omega, --eta and --domega must be given together")

    solver_kw = dict(max_iter=args.max_iter, tol_E=args.tol_E, tol_mat=args.tol_mat, mix=args.mix,
                     eq_time=args.eq_time, sqmat_shift=args.sqmat_shift,
                     round_fcidump=not args.no_fcidump_rounding)

    if args.U is not None or args.Ng is not None:
        if args.U is None or args.Ng is None:
            parser.error("--U and --Ng must be given together")
        if args.Ng % 2 != 0:
            parser.error("Ng must be even")

        Neff = 1 + args.Ng

        Rg0 = Rg1 = None
        if os.path.isfile(args.R_file):
            Rg0, Rg1 = read_R_guess(args.R_file)
            print(f"[guess] read R from {args.R_file}: atom0={Rg0}  atom1={Rg1}")

        lamg0 = lamg1 = None
        if os.path.isfile(args.lam_file):
            lamg0, lamg1 = read_lambda_guess(args.lam_file)
            print(f"[guess] read lambda from {args.lam_file}:\n  atom0:\n{lamg0}\n  atom1:\n{lamg1}")

        if Rg0 is not None and (len(Rg0) != Neff or len(Rg1) != Neff):
            parser.error(f"{args.R_file} rows must have {Neff} entries (Neff=1+Ng)")
        if lamg0 is not None and (lamg0.shape != (Neff, Neff) or lamg1.shape != (Neff, Neff)):
            parser.error(f"{args.lam_file} blocks must be {Neff}x{Neff}")

        res = run_gga(args.Ng, args.U, t=args.t, verbose=not args.quiet,
                       Rg0=Rg0, Rg1=Rg1, lamg0=lamg0, lamg1=lamg1, **solver_kw)
        print(f"Ng={res['Ng']}  U/t={res['U']/args.t:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
              f"E_var/t={res['E_var']/args.t:.6f}  docc={res['docc']:.6f}  iters={res['iters']}  "
              f"converged={res['converged']}")
        save_results(res)

        if args.omega is not None:
            omega_grid, A = compute_spectral_function(
                res["R0"], res["R1"], res["lam0"], res["lam1"], args.t,
                args.omega, args.eta, args.domega)
            save_spectral_function(omega_grid, A)
    else:
        t = args.t
        print(f"{'Ng':>4}{'U/t':>6}{'Z0':>10}{'Z1':>10}{'E_var/t':>12}{'docc':>10}")
        for Ng in [0, 2, 4]:
            for U in [0.0, 1.0, 2.0, 4.0, 8.0]:
                res = run_gga(Ng, U, t=t, **solver_kw)
                print(f"{Ng:4d}{U/t:6.2f}{res['Z0']:10.4f}{res['Z1']:10.4f}{res['E_var']/t:12.6f}{res['docc']:10.4f}")
