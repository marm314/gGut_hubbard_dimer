"""Exact diagonalization of the 2-site Hubbard dimer (4 orbitals: 2 sites x 2 spins).

H = -t * sum_sigma (c0s^dag c1s + h.c.) + U * sum_i n_{i,up} n_{i,dn} - mu * sum_i (n_{i,up}+n_{i,dn})

Half filling enforced via mu = U/2. Fermion operators are built explicitly via
Jordan-Wigner strings over the 4-mode local Fock space (dim = 2**4 = 16).

Ground-truth benchmark for the ghost Gutzwiller approximation (gGA) implementation.
"""

import numpy as np

# Mode ordering: 0 = site0-up, 1 = site0-dn, 2 = site1-up, 3 = site1-dn
N_MODES = 4
DIM = 2 ** N_MODES

I2 = np.eye(2)
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
C_LOCAL = np.array([[0.0, 1.0], [0.0, 0.0]])  # c|1> = |0>, c|0> = 0


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_c_ops(n_modes):
    """Jordan-Wigner annihilation operators c_i, i = 0..n_modes-1."""
    ops = []
    for i in range(n_modes):
        mats = []
        for j in range(n_modes):
            if j < i:
                mats.append(Z)
            elif j == i:
                mats.append(C_LOCAL)
            else:
                mats.append(I2)
        ops.append(_kron_list(mats))
    return ops


def build_hamiltonian(t, U, mu):
    c = build_c_ops(N_MODES)
    cdag = [op.conj().T for op in c]

    c0u, c0d, c1u, c1d = c
    c0u_d, c0d_d, c1u_d, c1d_d = cdag

    hop = -t * (c0u_d @ c1u + c1u_d @ c0u + c0d_d @ c1d + c1d_d @ c0d)

    n0u = c0u_d @ c0u
    n0d = c0d_d @ c0d
    n1u = c1u_d @ c1u
    n1d = c1d_d @ c1d

    interaction = U * (n0u @ n0d + n1u @ n1d)
    chem_pot = -mu * (n0u + n0d + n1u + n1d)

    H = hop + interaction + chem_pot
    n_ops = {"n0u": n0u, "n0d": n0d, "n1u": n1u, "n1d": n1d}
    return H, n_ops


def solve(t, U):
    mu = U / 2.0
    H, n_ops = build_hamiltonian(t, U, mu)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    E0 = evals[0]

    n_total = sum(gs.conj() @ (op @ gs) for op in n_ops.values()).real
    docc0 = (gs.conj() @ (n_ops["n0u"] @ n_ops["n0d"] @ gs)).real
    docc1 = (gs.conj() @ (n_ops["n1u"] @ n_ops["n1d"] @ gs)).real
    docc = 0.5 * (docc0 + docc1)

    return E0, n_total, docc


if __name__ == "__main__":
    t = 1.0
    U_values = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]

    print(f"{'U/t':>8}{'E0/t':>14}{'<n>':>10}{'docc/site':>12}")
    for U in U_values:
        E0, n_total, docc = solve(t, U)
        print(f"{U/t:8.2f}{E0/t:14.6f}{n_total:10.4f}{docc:12.4f}")
