"""Exact diagonalization of the 2-site Hubbard dimer WITH an extra site-splitting potential
Delta V, and its exact (Lehmann-representation) spectral function A(omega).

    H = -t sum_sigma (c0s^dag c1s + h.c.) + U sum_i n_{i,up} n_{i,dn}
        - mu sum_i (n_{i,up}+n_{i,dn}) + (Delta V / 2) (n_0 - n_1),      mu = U/2,

i.e. site 0 is raised by Delta V/2 and site 1 lowered by Delta V/2 relative to the plain
dimer of exact_dimer.py (Delta V = 0 recovers it exactly). Built the same way as
exact_dimer.py: explicit Jordan-Wigner fermion operators over the 4-mode (2 sites x 2 spins)
local Fock space (dim = 16); full dense diagonalization (trivial at this size), no symmetry
sectors assumed a priori.

The exact single-particle Green's function poles and weights are obtained directly from
the Lehmann representation,

    A(omega) = sum_{I,sigma} [ sum_m |<m,N+1| c^dag_{I sigma} |0,N>|^2 delta(omega - (E_m^{N+1}-E_0^N))
                               + sum_m |<m,N-1| c_{I sigma} |0,N>|^2 delta(omega + (E_0^N-E_m^{N-1})) ],

with |0,N> the N=2 (half-filled) ground state and the sums over ALL eigenstates of the
N+1=3 and N-1=1 sectors (not restricted to a closed-shell/symmetry subspace), each
summed over both sites and spins. At Delta V = 0 this reduces to the closed-form result of
/home/marm314/C_MEJUTO/gGut_notes/gGut/hubbard_dimer/gGut_notes/gGut.tex Eq. (eq:Aw_exact):
4 poles at +-(Omega/2-+t), Omega = sqrt(U^2+16t^2), weights 1+-4t/Omega -- used here only as
a validation check (see validate_against_closed_form), not assumed.
"""
import numpy as np

N_MODES = 4
DIM = 2 ** N_MODES
WEIGHT_TOL = 1e-10   # poles with total weight below this are dropped

I2 = np.eye(2)
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
C_LOCAL = np.array([[0.0, 1.0], [0.0, 0.0]])


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_c_ops(n_modes):
    ops = []
    for i in range(n_modes):
        mats = [Z if j < i else (C_LOCAL if j == i else I2) for j in range(n_modes)]
        ops.append(_kron_list(mats))
    return ops


def build_hamiltonian(t, U, deltaV):
    """Mode order: 0=site0-up, 1=site0-dn, 2=site1-up, 3=site1-dn."""
    mu = U / 2.0
    c = build_c_ops(N_MODES)
    cdag = [op.conj().T for op in c]
    c0u, c0d, c1u, c1d = c
    c0ud, c0dd, c1ud, c1dd = cdag

    hop = -t * (c0ud @ c1u + c1ud @ c0u + c0dd @ c1d + c1dd @ c0d)
    n0u, n0d, n1u, n1d = c0ud @ c0u, c0dd @ c0d, c1ud @ c1u, c1dd @ c1d
    n0, n1 = n0u + n0d, n1u + n1d

    H = hop + U * (n0u @ n0d + n1u @ n1d) - mu * (n0 + n1) + 0.5 * deltaV * (n0 - n1)
    return H, c, n0, n1


def eigensectors(H, n0, n1):
    """Block-diagonalize H by total particle number N. N is diagonal in the computational
    (occupation-number) basis itself -- each basis index's bit count IS its N -- so the
    sectors are split BEFORE diagonalizing (not after): at Delta V = 0, N=1 and N=3 are
    exactly degenerate (particle-hole symmetry at mu = U/2), and diagonalizing the full
    16x16 H at once lets eigh return eigenvectors that mix the two degenerate sectors."""
    basis_N = np.array([bin(k).count("1") for k in range(DIM)])
    sectors = {}
    for Nval in range(N_MODES + 1):
        idx = np.flatnonzero(basis_N == Nval)
        Hsec = H[np.ix_(idx, idx)]
        evals, evecs_sec = np.linalg.eigh(Hsec)
        evecs = np.zeros((DIM, len(idx)), dtype=H.dtype)
        evecs[idx, :] = evecs_sec
        sectors[Nval] = (evals, evecs)
    return sectors


def spectral_poles(t, U, deltaV):
    """Return (omega, weight) for every pole with weight > WEIGHT_TOL, sorted by omega."""
    H, c, n0, n1 = build_hamiltonian(t, U, deltaV)
    sec = eigensectors(H, n0, n1)

    E2, V2 = sec[2]
    assert len(E2) > 0, "empty N=2 sector"
    E0 = E2[0]
    psi0 = V2[:, 0]

    poles = []
    # addition: N=2 -> N=3
    E3, V3 = sec[3]
    for m in range(len(E3)):
        w = sum(np.abs(V3[:, m].conj() @ (op.conj().T @ psi0)) ** 2 for op in c)
        if w > WEIGHT_TOL:
            poles.append((E3[m] - E0, w))
    # removal: N=1 -> N=2 (hole added going from N=2 down to N=1)
    E1, V1 = sec[1]
    for m in range(len(E1)):
        w = sum(np.abs(V1[:, m].conj() @ (op @ psi0)) ** 2 for op in c)
        if w > WEIGHT_TOL:
            poles.append((E0 - E1[m], w))

    # merge poles at (near-)degenerate omega (e.g. the spin degeneracy of each N=1/N=3
    # level: two states of a spin doublet sit at the same energy, each contributing part
    # of the total weight at that single physical pole)
    poles.sort(key=lambda x: x[0])
    merged = []
    for om, w in poles:
        if merged and abs(om - merged[-1][0]) < 1e-7:
            merged[-1] = (merged[-1][0], merged[-1][1] + w)
        else:
            merged.append((om, w))
    poles = [(om, w) for om, w in merged if w > WEIGHT_TOL]
    return poles, E0


def validate_against_closed_form(t=1.0, Us=(0.5, 2.0, 5.0, 10.0), verbose=True):
    """Delta V = 0 check against gGut.tex Eq. (eq:Aw_exact)."""
    ok = True
    for U in Us:
        poles, E0 = spectral_poles(t, U, 0.0)
        Omega = np.sqrt(U ** 2 + 16.0 * t ** 2)
        inner, outer = Omega / 2.0 - t, Omega / 2.0 + t
        w_in, w_out = 1.0 + 4.0 * t / Omega, 1.0 - 4.0 * t / Omega
        expected = sorted([(-outer, w_out), (-inner, w_in), (inner, w_in), (outer, w_out)])
        got = sorted(poles)
        err_w = max(abs(g[0] - e[0]) for g, e in zip(got, expected))
        err_omega = max(abs(g[1] - e[1]) for g, e in zip(got, expected))
        E0_exact = -U / 2.0 - Omega / 2.0
        err_E0 = abs(E0 - E0_exact)
        if verbose:
            print(f"  U/t={U:5.2f}: n_poles={len(got)} (expected 4), max|domega|={err_w:.2e}, "
                  f"max|dweight|={err_omega:.2e}, |E0-E0_exact|={err_E0:.2e}")
        ok &= (len(got) == 4 and err_w < 1e-8 and err_omega < 1e-8 and err_E0 < 1e-8)
    return ok


if __name__ == "__main__":
    print("validating Delta V = 0 against the closed-form result of gGut.tex:")
    ok = validate_against_closed_form()
    print("OK" if ok else "FAILED")
