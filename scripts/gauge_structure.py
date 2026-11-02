"""What produces the gauge: singular values of the constraint Jacobian J_Phi, the natural orbitals of ^cDelta (occupations), and the null directions of J_Phi^T written as shifts
of the multiplier matrix lambda_c in the natural-orbital basis. Also: are the pure natural orbitals exact eigenvectors of rho and of H_qp?
GHOST_EPS=1e-7 PYTHONPATH=scripts python3 scripts/gauge_structure.py"""
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
np.set_printoptions(linewidth=200, suppress=True)
for V, tc in ((1.4142135623730951, 1.0), (0.7071067811865476, 1.0), (1.4142135623730951, 3.0)):
    s7 = np.load(f"cederbaum_data/scan_ghost_tc_eps1e-07/V{V}.npz")
    for B in (3, 5):
        z = s7[f"tc{tc}_B{B}_z"]
        if B == 3: m = g.Model(1.0, 10.0, V, tc, 3, 1e-7, "cv"); P = N.Problem(m)
        else: P, _ = embed(s7[f"tc{tc}_B3_z"], V, tc, 1e-7); m = P.m
        E, gz, st, _ = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; rho = st["rho"]; Dl = st["Dl"][0]
        U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]
        n, W = np.linalg.eigh(Dl); sl = m.sl[0]
        print(f"\nV/g={V:.3f} t_c/g={tc} B_c={B}: J_Phi is {J.shape[0]} x {J.shape[1]}; singular values (descending, last 8): {np.array2string(s[-8:][::-1], precision=2)}; rank {rank}; free directions {null.shape[1]}")
        print(f"  natural orbitals of ^c Delta (occupations): {np.array2string(n, precision=7)}")
        # constraint components of block c (rows 0..B(B+1)/2-1): lambda_aa = -mu_aa/2, lambda_ab = -mu_ab/4
        for j in range(null.shape[1]):
            mu = null[:, j]; lam = np.zeros((B, B)); row = 0
            for a, b in zip(*P.iu[0]): lam[a, b] = lam[b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
            rest = np.linalg.norm(mu[row:])
            L = W.T @ lam @ W; off = np.linalg.norm(L - np.diag(np.diag(L)))
            print(f"  null direction {j}: weight on o,v rows {rest:.1e}; delta-lambda_c in the natural basis: diagonal {np.array2string(np.diag(L), precision=3)}, off-diagonal norm {off:.1e}")
        # pure natural orbitals: are they eigenvectors of rho and of H_qp?
        for k in range(B):
            u = np.zeros(m.n); u[sl] = W[:, k]; rr = np.linalg.norm(rho @ u - n[k].round() * u) if min(abs(n[k]), abs(1 - n[k])) < 1e-4 else np.nan
            print(f"    natural orbital {k}: occupation {n[k]:.7f}; |rho u - round(n) u| = {rr:.1e}" if not np.isnan(rr) else f"    natural orbital {k}: occupation {n[k]:.7f} (not pure)")


# ---- second part: exact singular values and the block structure of the null directions (V/g = sqrt2, t_c/g = 1)
print("\n=== exact singular values and block structure of the null space (V/g=sqrt2, t_c/g=1) ===")
V, tc = 1.4142135623730951, 1.0; s7 = np.load(f"cederbaum_data/scan_ghost_tc_eps1e-07/V{V}.npz")
for B in (3, 5):
    z = s7[f"tc{tc}_B{B}_z"]
    if B == 3: m = g.Model(1.0, 10.0, V, tc, 3, 1e-7, "cv"); P = N.Problem(m)
    else: P, _ = embed(s7[f"tc{tc}_B3_z"], V, tc, 1e-7); m = P.m
    E, gz, st, _ = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; Dl = st["Dl"][0]; n, W = np.linalg.eigh(Dl)
    U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]
    print(f"B_c={B}: all singular values of J_Phi: {np.array2string(s, precision=2, formatter={'float_kind': lambda x: f'{x:.1e}'})}; cut 1e-5*s_max = {1e-5*s[0]:.1e}")
    pure = [k for k in range(B) if min(n[k], 1 - n[k]) < 1e-6]; filled = [k for k in pure if n[k] > 0.5]; empty = [k for k in pure if n[k] < 0.5]
    print(f"  pure natural orbitals: empty {empty}, filled {filled}")
    blocks = {"empty-empty": [(a, a) for a in empty], "filled-filled": [(a, b) for a in filled for b in filled if a <= b], "empty-filled": [(a, b) for a in empty for b in filled]}
    for j in range(null.shape[1]):
        mu = null[:, j]; lam = np.zeros((B, B)); row = 0
        for a, b in zip(*P.iu[0]): lam[a, b] = lam[b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
        L = W.T @ lam @ W; tot = np.linalg.norm(L); inb = {}
        mask = np.zeros((B, B), bool)
        for name, lst in blocks.items():
            mm = np.zeros((B, B), bool)
            for a, b in lst: mm[a, b] = mm[b, a] = True
            inb[name] = np.linalg.norm(L[mm]) / tot
            if name != "empty-filled": mask |= mm
        print(f"  null direction {j}: fraction of |delta-lambda_c|: empty-empty {inb['empty-empty']:.3f}, filled-filled {inb['filled-filled']:.3f}, empty-filled {inb['empty-filled']:.1e}; outside the pure blocks {np.linalg.norm(L[~mask])/tot:.1e}")
    # rows of J_Phi that are (numerically) zero: the constraint components in the natural basis
    row = 0; Jn = []
    for a, b in zip(*P.iu[0]): row += 1
    ncore = row; Jc = J[:ncore, :]
    # transform the constraint components to the natural basis: kappa'_{kl} = sum w_ka w_lb kappa_ab
    full = np.zeros((B, B, J.shape[1])); r = 0
    for a, b in zip(*P.iu[0]):
        full[a, b] = Jc[r]; full[b, a] = Jc[r]; r += 1
    nat = np.einsum("ak,bl,abj->klj", W, W, full)
    for k in range(B):
        for l in range(k, B):
            nrm = np.linalg.norm(nat[k, l]); typ = ("empty" if n[k] < 1e-6 else "filled" if n[k] > 1 - 1e-6 else "mixed") + "-" + ("empty" if n[l] < 1e-6 else "filled" if n[l] > 1 - 1e-6 else "mixed")
            print(f"    row (k={k},l={l}) [{typ}]: |gradient of kappa_kl w.r.t. Phi| = {nrm:.1e}")
