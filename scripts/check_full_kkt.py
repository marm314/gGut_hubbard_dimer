"""Full KKT test at eps=1e-9 solutions: multipliers from the Phi part only vs from all variables; residuals and ||[H_qp,rho]||. PYTHONPATH=scripts python3 scripts/check_full_kkt.py"""
import numpy as np, sys
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
def run(V, tc, Bc):
    data = np.load(f"cederbaum_data/scan_ghost_tc/V{V}.npz"); z = data[f"tc{tc}_B{Bc}_z"]
    if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
    else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc); m = P.m
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z); nx = P.nx
    o = N.recover(P, z)
    mu_f, res, rk, sv = np.linalg.lstsq(J.T, -gz, rcond=None)
    r_full = np.linalg.norm(J.T @ mu_f + gz); r_phi_only = np.linalg.norm(J[:, :nx].T @ mu_f + gz[:nx]); r_X = np.linalg.norm(J[:, nx:].T @ mu_f + gz[nx:])
    mu_p, *_ = np.linalg.lstsq(J[:, :nx].T, -gz[:nx], rcond=None)
    rX_p = np.linalg.norm(J[:, nx:].T @ mu_p + gz[nx:])
    def comm_of(mu):
        lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for a, b in zip(*P.iu[I]):
                lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
        H = m.build_Hqp(o["R"], lam, Tb, u, w, o["nu"]); return np.linalg.norm(H @ o["rho"] - o["rho"] @ H)
    print(f"V={V:.3f} tc={tc} B_c={Bc}: |grad E| phi {np.linalg.norm(gz[:nx]):.1e} X {np.linalg.norm(gz[nx:]):.1e}; |cons| {np.abs(P.constraints(z)).max():.1e}")
    print(f"   mu from phi-only lstsq: resid phi {np.linalg.norm(J[:, :nx].T @ mu_p + gz[:nx]):.1e}, resid X {rX_p:.1e}, comm {comm_of(mu_p):.1e}")
    print(f"   mu from FULL lstsq    : resid phi {r_phi_only:.1e}, resid X {r_X:.1e}, total {r_full:.1e}, comm {comm_of(mu_f):.1e}  (rank J {rk}/{J.shape[0]}, smallest sv {sv[-1]:.1e})")
for V, tc, Bc in [(1.4142135623730951, 1.0, 3), (1.4142135623730951, 1.0, 5), (0.7071067811865476, 1.0, 3), (1.4142135623730951, 4.6, 5), (0.7071067811865476, 0.2, 3)]: run(V, tc, Bc)
