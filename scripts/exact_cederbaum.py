"""Exact diagonalization of the Cederbaum model (+ weak core--occupied hopping H_t).

H   = Delta sum_s (n_vs - n_os) - E_c sum_s n_cs
      - V sum_{s,s'} (1 - n_cs) (c^dag_vs' c_os' + h.c.)
H_t = t_c sum_s (c^dag_cs c_os + h.c.)

Three orbitals (c, o, v) x two spins = 6 modes, Fock space dim 2**6 = 64.
Mode ordering is spin-major: 0,1,2 = c,o,v (up); 3,4,5 = c,o,v (down), i.e. all
up operators to the left of all down operators, as assumed for the sign
conventions of Tab. I in main.tex (only the labels of the singlet/triplet
combinations depend on it; energies and weights do not).

Hole Green's function of the core orbital (main.tex, Eq. hc_GF), for sigma = up:

    G^h_cc(w) = sum_n |<n| c_{c,up} |GS>|^2 / (w + (E_n - E_GS) - i eta),

with poles at w = -(E_n - E_GS) = -E_i and weights p_i.

Reuses the Jordan-Wigner builder of gga_dimer.py.
"""

import numpy as np

from gga_dimer import build_c_ops

C_UP, O_UP, V_UP, C_DN, O_DN, V_DN = range(6)


def build_hamiltonian(Delta, Ec, V, tc=0.0, ht="co"):
    """ht: 'co' = t_c(c^dag_c c_o + h.c.) (main.tex); 'cv' = t_c(c^dag_c c_v + h.c.)."""
    c = [op.toarray() for op in build_c_ops(6)]
    cd = [op.T for op in c]
    n = [cd[i] @ c[i] for i in range(6)]
    one = np.eye(64)

    H = np.zeros((64, 64))
    for cc, oo, vv in ((C_UP, O_UP, V_UP), (C_DN, O_DN, V_DN)):
        H += Delta * (n[vv] - n[oo]) - Ec * n[cc]
        tgt = oo if ht == "co" else vv
        H += tc * (cd[cc] @ c[tgt] + cd[tgt] @ c[cc])
    for cc in (C_UP, C_DN):
        for oo, vv in ((O_UP, V_UP), (O_DN, V_DN)):
            hop = cd[vv] @ c[oo] + cd[oo] @ c[vv]
            H -= V * (one - n[cc]) @ hop
    return H, c, n


def solve(Delta, Ec, V, tc=0.0, tol_deg=1e-9, tol_w=1e-12, ht="co"):
    """Ground state and hole spectrum of G_cc.

    Returns dict with E0, the number of (near-)degenerate ground states, the
    ground-state core occupation nu_c = <n_c> and the poles/weights of G^h_cc
    (degenerate poles within tol_deg merged, weights below tol_w dropped)."""
    H, c, n = build_hamiltonian(Delta, Ec, V, tc, ht)
    E, U = np.linalg.eigh(H)
    E0 = E[0]
    ndeg = int(np.sum(E < E0 + tol_deg))
    gs = U[:, 0]

    amp = U.T @ (c[C_UP] @ gs)           # <n|c_{c,up}|GS> for every eigenstate n
    w = amp ** 2
    omega = -(E - E0)
    keep = w > tol_w
    omega, w = omega[keep], w[keep]
    order = np.argsort(-omega)           # ascending ionization energy
    omega, w = omega[order], w[order]

    poles, weights = [], []
    for o, p in zip(omega, w):
        if poles and abs(o - poles[-1]) < tol_deg:
            weights[-1] += p
        else:
            poles.append(o)
            weights.append(p)

    return {
        "E0": E0, "ndeg": ndeg,
        "nu_c": float(gs @ ((n[C_UP] + n[C_DN]) @ gs)),
        "rho_co": float(gs @ (c[C_UP].T @ c[O_UP]) @ gs), "rho_cv": float(gs @ (c[C_UP].T @ c[V_UP]) @ gs),
        "rho_vo": float(gs @ (c[V_UP].T @ c[O_UP]) @ gs),
        "poles": np.array(poles), "weights": np.array(weights),
    }


def closed_form(Delta, Ec, V):
    """Poles (-E_i) and strengths p_i of main.tex Eq. (Poles&Strengths), t_c = 0."""
    a = V / Delta
    s = np.sqrt(1.0 + a * a)
    E = np.array([Ec + 2 * Delta - 2 * Delta * s, Ec + 2 * Delta, Ec + 2 * Delta + 2 * Delta * s])
    p = np.array([(1 + s) ** 2 / (4 * (1 + a * a)),
                  a * a / (2 * (1 + a * a)),
                  (1 - s) ** 2 / (4 * (1 + a * a))])
    return -E, p


def noninteracting_energy(Delta, Ec, tc, ht="co"):
    """V = 0 energy in the 4-electron sector: two lowest levels of the 3x3 one-body problem
    (orbitals c, o, v), times 2 spins. ht selects the c-o or c-v hopping."""
    h = np.array([[-Ec, 0.0, 0.0], [0.0, -Delta, 0.0], [0.0, 0.0, Delta]])
    j = 1 if ht == "co" else 2
    h[0, j] = h[j, 0] = tc
    return 2.0 * np.sum(np.linalg.eigvalsh(h)[:2])


if __name__ == "__main__":
    Ec = 10.0
    print("t_c = 0: ED vs closed form (main.tex Eq. Poles&Strengths)")
    print(f"{'Delta':>6}{'V':>6}{'E0':>12}{'-2Ec-2D':>12}{'ndeg':>5}{'nu_c':>8}   pole(ED)  pole(exact)   p(ED)  p(exact)")
    worst = 0.0
    for Delta in (1.0, 0.5):
        for V in (0.5, 1.0, 2.0):
            r = solve(Delta, Ec, V, 0.0)
            pc, wc = closed_form(Delta, Ec, V)
            print(f"{Delta:6.2f}{V:6.2f}{r['E0']:12.6f}{-2*Ec-2*Delta:12.6f}{r['ndeg']:5d}{r['nu_c']:8.4f}")
            assert len(r["poles"]) == 3, r["poles"]
            for k in range(3):
                print(f"{'':37}{r['poles'][k]:11.6f}{pc[k]:11.6f}{r['weights'][k]:9.6f}{wc[k]:9.6f}")
            worst = max(worst, np.abs(r["poles"] - pc).max(), np.abs(r["weights"] - wc).max(),
                        abs(r["E0"] + 2 * Ec + 2 * Delta))
    print(f"max |ED - closed form| = {worst:.2e}")
