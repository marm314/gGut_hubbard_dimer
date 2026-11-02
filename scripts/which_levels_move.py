"""Which levels of H_qp can be moved by the gauge shift? For random gauge parameters a (|a| ~ 30) compare the sorted-by-eigenvector levels with those at a=0:
a level 'moves' if its eigenvalue changes (eigenvectors matched by overlap). Also block weights (core / o / v) and the weights |R_I^T phi^(I)|^2 of each level.
PYTHONPATH=scripts python3 scripts/which_levels_move.py"""
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
np.set_printoptions(linewidth=200, suppress=True)
def build(V, tc, B, eps, scan):
    z = scan[f"tc{tc}_B{B}_z"]
    if B == 3: m = g.Model(1.0, 10.0, V, tc, 3, eps, "cv"); P = N.Problem(m)
    else: P, _ = embed(scan[f"tc{tc}_B3_z"], V, tc, eps); m = P.m
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
    def H(a):
        mu = mu0 + null @ a; lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for i, j in zip(*P.iu[I]): lam[I][i, j] = lam[I][j, i] = (-mu[row] / 2.0 if i == j else -mu[row] / 4.0); row += 1
        return m.build_Hqp(st["R"], lam, Tb, u, w, o["nu"])
    return m, st["R"], H, null.shape[1], st["rho"]
for eps, SUF, cases in ((1e-9, "_eps1e-09", ((0.7071067811865476, 0.1), (1.4142135623730951, 0.05))), (1e-7, "_eps1e-07", ((1.4142135623730951, 1.0),))):
    for V, tc in cases:
        scan = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
        for B in (3, 5):
            m, R, H, na, rho = build(V, tc, B, eps, scan); H0 = H(np.zeros(na)); e0, ph0 = np.linalg.eigh(H0); occ = np.einsum("ik,ij,jk->k", ph0, rho, ph0)
            rng = np.random.default_rng(0); moved = np.zeros(len(e0)); a = rng.standard_normal(na) * 30; e1, ph1 = np.linalg.eigh(H(a))
            # match levels by eigenvector overlap (core-block ghost orbitals keep their vectors; compare by max overlap)
            ov = np.abs(ph0.T @ ph1); match = ov.argmax(1); d = np.array([abs(e1[match[k]] - e0[k]) for k in range(len(e0))]); keep = ov.max(1)
            print(f"\nV/g={V:.3f} t_c/g={tc} B_c={B} (eps={eps:g}, {na} free directions): level | occ | weight on core/o/v block | |R^T phi|^2 (c,o,v) | change under a random shift (|a|~{np.linalg.norm(a):.0f}) | eigenvector kept (overlap)")
            for k in range(len(e0)):
                blk = [(ph0[m.sl[I], k] ** 2).sum() for I in range(3)]; W = [(R[I] @ ph0[m.sl[I], k]) ** 2 for I in range(3)]
                print(f"   {e0[k]:9.3f} | {int(round(occ[k]))} | {blk[0]:.3f} {blk[1]:.3f} {blk[2]:.3f} | {W[0]:.1e} {W[1]:.1e} {W[2]:.1e} | {d[k]:9.3f} | {keep[k]:.4f}  {'MOVES' if d[k] > 1e-3 else 'fixed'}")
