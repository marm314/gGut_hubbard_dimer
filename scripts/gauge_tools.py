"""Helpers: the gGut solution of the eps=1e-7 scan at a given gauge of the multipliers (min-norm gauge, or the gauge in which the ghost poles are placed on the
ED satellites, cederbaum_data/ghost_placed). Used by the spectral-function figures."""
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
from bc5_cederbaum import embed

EPS = 1e-7


def solution_at_gauge(V, tc, Bc, placed=True, eps=EPS):
    """Returns dict(eps, phi, W[I,k]=(R_I^T phi_k^(I))^2, Nocc, comm, R, a) of the stored solution; the multipliers are mu_0 + N a with a = a* (placed) or 0."""
    scan = np.load(f"cederbaum_data/scan_ghost_tc_eps{eps:g}/V{V}.npz"); z = scan[f"tc{tc}_B{Bc}_z"]
    if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, eps, "cv"); P = N.Problem(m)
    else: P, _ = embed(scan[f"tc{tc}_B3_z"], V, tc, eps); m = P.m
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
    a = np.load("cederbaum_data/ghost_placed" + ("" if eps == 1e-7 else f"_eps{eps:g}") + f"/V{V}.npz")[f"tc{tc}_B{Bc}_a"] if placed else np.zeros(null.shape[1])
    mu = mu0 + null @ a; lam = [np.zeros((b, b)) for b in m.B]; row = 0
    for I in range(3):
        for i, j in zip(*P.iu[I]): lam[I][i, j] = lam[I][j, i] = (-mu[row] / 2.0 if i == j else -mu[row] / 4.0); row += 1
    rho, R = st["rho"], st["R"]; H = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, ph = np.linalg.eigh(H)
    W = np.array([[(R[I] @ ph[m.sl[I], k]) ** 2 for k in range(e.size)] for I in range(3)])
    occ = np.einsum("ik,ij,jk->k", ph, rho, ph)
    return dict(eps=e, phi=ph, W=W, Nocc=m.Nocc, comm=np.linalg.norm(H @ rho - rho @ H), R=R, a=a, E=E, rho=rho, occ=occ)
