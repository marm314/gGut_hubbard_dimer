"""G0W0 for the three-level core-hole model, compared with ED (t_c=0).

Model of Cederbaum, Domcke, Schirmer (PRA 22, 206) / Kocklauner & Golze (arXiv:2412.17132, Fig. 2): the core hole
couples to ONE valence excitation of energy Omega = eps_v - eps_o with transition moment rho. In our notation
Omega = 2g and rho = sqrt(2) V (the o->v excitation of two spin channels, normalized, couples with sqrt(2) V: the
off-diagonal element of Eq. (28)-(29) of Cederbaum 1980). The check of rho = sqrt(2) V is the small-V limit of the
exact satellite/QP weight ratio, which must be gamma = rho^2/Omega^2 (GW+C excitation weight).

G0W0 (G0 = bare core level eps_c = -E_c, W = one bosonic mode):
    Sigma_c(w) = rho^2 / (w - eps_c + Omega)        (core line occupied; boson emitted)
    G(w) = 1/(w - eps_c - Sigma_c(w)), poles x = w - eps_c solve x (x + Omega) = rho^2,
    weights Z = 1/(1 + rho^2/(x + Omega)^2).
No exchange / Hedin shift (none is defined for this model). Usage: python3 g0w0_cederbaum.py"""
import numpy as np
import exact_cederbaum as ed

G, EC = 1.0, 10.0
OMEGA = 2.0 * G


def g0w0(rho, Omega=OMEGA, Ec=EC):
    x = np.array([-Omega / 2 + np.sqrt(Omega ** 2 / 4 + rho ** 2), -Omega / 2 - np.sqrt(Omega ** 2 / 4 + rho ** 2)])
    Z = 1.0 / (1.0 + rho ** 2 / (x + Omega) ** 2)
    return -Ec + x, Z


def gwc(rho, Omega=OMEGA, nmax=6):
    """GW+C: QP at the G0W0 QP pole, Poisson satellites at -n Omega with weights Z gamma^n e^-gamma / n! (relative)."""
    from math import factorial
    gam = (rho / Omega) ** 2
    n = np.arange(nmax)
    return -n * Omega, np.exp(-gam) * gam ** n / np.array([factorial(k) for k in n])


def exact(V):
    r = ed.solve(G, EC, V, 0.0, ht="cv")
    return r["poles"], r["weights"]


if __name__ == "__main__":
    print("rho = sqrt(2) V check: exact (satellite-1 weight)/(QP weight) vs gamma = rho^2/Omega^2 at small V")
    for V in (0.02, 0.05, 0.1):
        p, w = exact(V)
        print(f"  V={V}: ratio {w[1]/w[0]:.6f}   (sqrt2 V)^2/Omega^2 = {2*V**2/OMEGA**2:.6f}   V^2/Omega^2 = {V**2/OMEGA**2:.6f}")
    print("\nPositions relative to the QP pole, in units of Omega; weights; G0W0 vs exact (t_c=0)")
    for lab, V in [("rho/Omega=0.5", 0.5 * OMEGA / np.sqrt(2)), ("rho/Omega=1.0", 1.0 * OMEGA / np.sqrt(2))]:
        rho = np.sqrt(2) * V
        pe, we = exact(V)
        pg, zg = g0w0(rho)
        print(f"\n{lab}: V/g={V:.4f}, rho/Omega={rho/OMEGA:.4f}")
        print("  exact  :", np.round((pe - pe[0]) / OMEGA, 3), np.round(we, 3), " sum", round(we.sum(), 4))
        print("  G0W0   :", np.round((pg - pg[0]) / OMEGA, 3), np.round(zg, 3), " sum", round(zg.sum(), 4))
        pc, wc = gwc(rho)
        print("  GW+C   :", np.round(pc / OMEGA, 3)[:4], np.round(wc, 3)[:4])
        print("  abs. QP pole: exact %.4f  G0W0 %.4f" % (pe[0], pg[0]))
