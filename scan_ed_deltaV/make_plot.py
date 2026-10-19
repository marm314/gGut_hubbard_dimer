"""ED pole positions vs U/t for the half-filled Hubbard dimer with and without a site-
splitting potential Delta V, using the validated closed-form solution of ED_deltaV.tex
(cross-checked to machine precision against exact_dimer_deltaV.py's numerical exact
diagonalization; see validate() below and exact_dimer_deltaV.validate_against_closed_form
for Delta V = 0).

Writes eqp_vs_U.png (this directory) and eqp_vs_U.txt (U, then the 4 Delta V=0 poles,
then the 4 Delta V=2 poles, then the corresponding 8 weights).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import exact_dimer_deltaV as e

t = 1.0
DELTA_V_COMPARE = 2.0


def H2_explicit(t, U, deltaV):
    """H^(2) (Eq. eq:H2 of ED_deltaV.tex) with every entry substituted to be an explicit
    elementary function of t, U, Delta V (Eqs. eq:cos4theta, eq:H2-explicit-diag/mixed/offdiag)
    -- no numerically diagonalized theta anywhere."""
    xi2 = t ** 2 + (deltaV / 2.0) ** 2
    xi = np.sqrt(xi2)
    bracket = (deltaV / 2.0) ** 2 - t ** 2  # = xi^2 cos(4 theta)
    Hgg = -U / 4 - 2 * xi + U * bracket / (4 * xi2)
    Huu = -U / 4 + 2 * xi + U * bracket / (4 * xi2)
    Hgu = U / 4 - U * bracket / (4 * xi2)
    HOO = -U / 2 - U * bracket / (2 * xi2)
    HgO = np.sqrt(2) * U * t * deltaV / (4 * xi2)
    HuO = -np.sqrt(2) * U * t * deltaV / (4 * xi2)
    M = np.array([[Hgg, Hgu, HgO], [Hgu, Huu, HuO], [HgO, HuO, HOO]])
    return M, xi


def E0_explicit(t, U, deltaV):
    """Eq. eq:E0-explicit of ED_deltaV.tex: the k=2 branch of the trigonometric cubic
    (verified there, by a 2000-point numerical scan, to always be the smallest root) --
    a single explicit function of t, U, Delta V, via p, q of Eq. eq:pq-explicit."""
    xi2 = t ** 2 + (deltaV / 2.0) ** 2
    p = -4 * xi2 - U ** 2 / 3
    q = 2 * U ** 3 / 27 + 4 * U * xi2 / 3 - U * deltaV ** 2
    arg = np.clip((3 * q) / (2 * p) * np.sqrt(-3 / p), -1, 1)
    return 2 * np.sqrt(-p / 3) * np.cos((1 / 3) * np.arccos(arg) - 4 * np.pi / 3) - U / 3


def closed_form_eig(t, U, deltaV):
    """E0 and (a_g,a_u,a_O) purely from t, U, Delta V via Eqs. eq:H2-explicit-*,
    eq:E0-explicit, eq:a-explicit of ED_deltaV.tex (no eigh anywhere in the computation
    itself) -- cross-checked against np.linalg.eigh every call as a safety net."""
    M, xi = H2_explicit(t, U, deltaV)
    E0 = E0_explicit(t, U, deltaV)
    if abs(deltaV) < 1e-12:
        # Delta V = 0: Eq. eq:a-explicit is a 0/0 indeterminate form there; use the known
        # reduction directly instead (Section "The N=2 sector" of ED_deltaV.tex)
        Omega = np.sqrt(U ** 2 + 16.0 * t ** 2)  # xi = t here
        v0 = np.array([np.sqrt((1 + 4 * t / Omega) / 2), -np.sqrt((1 - 4 * t / Omega) / 2), 0.0])
    else:
        b11, b12, b13 = M[0, 0] - E0, M[0, 1], M[0, 2]
        b22, b23 = M[1, 1] - E0, M[1, 2]
        ng = b12 * b23 - b13 * b22
        nu = b12 * b13 - b23 * b11
        nO = b11 * b22 - b12 ** 2
        v0 = np.array([ng, nu, nO])
        v0 = v0 / np.linalg.norm(v0)

    evals_check, evecs_check = np.linalg.eigh(M)
    assert abs(E0 - evals_check[0]) < 1e-8, "closed-form cubic disagrees with eigh"
    v_true = evecs_check[:, 0]
    if np.dot(v_true, v0) < 0:
        v_true = -v_true
    assert np.abs(v0 - v_true).max() < 1e-6, "closed-form eigenvector disagrees with eigh"
    return E0, v0, xi


def analytic_poles(t, U, deltaV):
    """Returns ([omega_1..4 sorted], [weight_1..4]), closed form validated against ED."""
    E0, (a0, a1, a2), xi = closed_form_eig(t, U, deltaV)
    mu = U / 2.0
    Wg, Wu = 2 * a0 ** 2 + a2 ** 2, 2 * a1 ** 2 + a2 ** 2
    omega_g, omega_u = E0 + mu + xi, E0 + mu - xi
    poles = sorted([(-omega_g, Wg), (-omega_u, Wu), (omega_u, Wu), (omega_g, Wg)])
    return [p[0] for p in poles], [p[1] for p in poles]


def validate(n=8, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        tt = rng.uniform(0.3, 2.0)
        UU = rng.uniform(0.1, 10.0)
        dV = rng.uniform(0.0, 4.0)
        om, w = analytic_poles(tt, UU, dV)
        num, _ = e.spectral_poles(tt, UU, dV)
        worst = max(worst, max(abs(om[i] - num[i][0]) for i in range(4)),
                    max(abs(w[i] - num[i][1]) for i in range(4)))
    return worst


if __name__ == "__main__":
    worst = validate()
    print(f"random cross-check vs numerical ED (8 random (t,U,DeltaV)): max error = {worst:.2e}")
    assert worst < 1e-8, "analytic closed form does not match numerical ED -- do not trust the plot"

    Us = np.round(np.arange(0.1, 10.01, 0.1), 1)
    rows = []
    for U in Us:
        om0, w0 = analytic_poles(t, U, 0.0)
        om2, w2 = analytic_poles(t, U, DELTA_V_COMPARE)
        rows.append([U] + om0 + om2 + w0 + w2)
    rows = np.array(rows)
    header = ("U/t  om0_1..4(DeltaV=0)  om0_1..4(DeltaV=" + str(DELTA_V_COMPARE) +
              ")  w_1..4(DeltaV=0)  w_1..4(DeltaV=" + str(DELTA_V_COMPARE) + ")")
    np.savetxt("eqp_vs_U.txt", rows, header=header)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    labels_done = set()
    for k, (lo, hi, name) in enumerate([(0, 1, "satellite"), (1, 2, "quasi-particle"),
                                        (2, 3, "quasi-particle"), (3, 4, "satellite")]):
        col = "tab:blue" if name == "quasi-particle" else "tab:orange"
        lab0 = f"ED, {name}, $\\Delta V=0$" if (name, 0) not in labels_done else None
        lab2 = f"ED, {name}, $\\Delta V={DELTA_V_COMPARE:g}$" if (name, 1) not in labels_done else None
        labels_done.add((name, 0)); labels_done.add((name, 1))
        ax.plot(Us, rows[:, 1 + lo], "--", color=col, lw=1.4, label=lab0)
        ax.plot(Us, rows[:, 5 + lo], "-", color=col, lw=1.6, label=lab2)

    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("U/t")
    ax.set_ylabel("pole energy $\\omega$ / t")
    ax.set_title("Exact (ED) spectral-function poles vs U:\n"
                  f"$\\Delta V=0$ (dashed) vs $\\Delta V={DELTA_V_COMPARE:g}$ (solid)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("eqp_vs_U.png", dpi=150)
    print("wrote eqp_vs_U.png and eqp_vs_U.txt")
