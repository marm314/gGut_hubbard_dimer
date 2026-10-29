"""Numbers for the 'absence of satellites' paragraph of doc/cederbaum.tex at V/g = 1/sqrt2, sqrt2 (t_c=0.1, B_c=3):
natural orbitals of ^c Delta, R_c along them, weights of the occupied qp levels, rank of the constraint Jacobian and the
direction it leaves undetermined, and the 'hand mixing' of the two filled core orbitals in lambda_c (weights, comm residual).
python3 scripts/analyze_absence_golze.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe

TC = 0.1
for V in pe.VS:
    m = g.Model(1.0, 10.0, V, TC, 3, 1e-9, "cv"); P = N.Problem(m); r = ed.solve(1.0, 10.0, V, TC, ht="cv")
    class _R: pass
    res = _R(); res.x = np.load("cederbaum_data/ghost_solutions.npz")[f"V{V}_tc{TC}_z3"]   # cached minimizer solution (verify_ghost_poles.py): the one behind Figs. 2 and 3
    res.fun = P.energy_and_grad(res.x)[0]
    o = N.recover(P, res.x); best = (res, o)
    print(f"\nV={V:.4f}  t_c={TC}: E-E_ED={float(res.fun)-r['E0']:.2e} kkt={o['kkt']:.1e} comm={o['comm']:.1e}", flush=True)
    Dc = o["Dl"][0] if "Dl" in o else None
    n, W = np.linalg.eigh(Dc)
    Rc = o["R"][0]
    print("  natural-orbital occupations of ^c Delta:", np.array2string(n, precision=3), " (eigh order)")
    print("  R_c components along them:", np.array2string(W.T @ Rc, precision=2), " R_c^T R_c =", round(float(Rc @ Rc), 4),
          " fraction along the largest:", round(float((W.T @ Rc)[np.argmax(np.abs(W.T @ Rc))] ** 2 / (Rc @ Rc)), 6))
    print("  occupied qp levels:", np.round(o["poles"], 3), "weights", np.array2string(o["weights"], precision=2))
    J = P.cons_jac(res.x)[:, :P.nx]
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-6 * sv[0]).sum()); print(f"  constraint Jacobian {J.shape}: rank {rank}; smallest singular values {sv[-3:]}")
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    nullm = U[:, rank:]          # multiplier directions with J^T n = 0
    print("  undetermined multiplier directions:", nullm.shape[1])
    mu0, *_ = np.linalg.lstsq(J.T, -P.energy_and_grad(res.x)[1][:P.nx], rcond=None)
    st = o["st"]; E_, gz, st_, (hyb, ls, Tb, u, w) = P.energy_and_grad(res.x)
    def levels(mu):
        lam = [np.zeros((b, b)) for b in m.B]; row = 0
        for I in range(3):
            for a, b in zip(*P.iu[I]):
                lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0); row += 1
        H = m.build_Hqp(o["R"], lam, Tb, u, w, o["nu"]); e, phi = np.linalg.eigh(H)
        return e, np.array([(o["R"][0] @ phi[m.sl[0], k]) ** 2 for k in range(m.Nocc)]), H
    e0, w0, H0 = levels(mu0)
    for j in range(nullm.shape[1]):
        e1, w1, _ = levels(mu0 + 1.0 * nullm[:, j])
        print(f"  shift along null direction {j}: levels change by {np.array2string(e1[:m.Nocc]-e0[:m.Nocc], precision=3)}; weights {np.array2string(w1, precision=2)}")
    # hand mixing of the two filled core orbitals in lambda_c (natural-orbital basis of ^c Delta)
    filled = np.where(n > 0.5)[0]
    print("  filled natural orbitals:", filled, n[filled])
    lam = [x.copy() for x in o["lam"]]
    for dlt in (0.0, 1.0, 3.0):
        lm = [x.copy() for x in lam]
        a, b = filled[:2]
        X = np.zeros_like(lm[0]); X += dlt * (np.outer(W[:, a], W[:, b]) + np.outer(W[:, b], W[:, a])); lm[0] = lm[0] + X
        H = m.build_Hqp(o["R"], lm, Tb, u, w, o["nu"]); e, phi = np.linalg.eigh(H)
        wt = np.array([(o["R"][0] @ phi[m.sl[0], k]) ** 2 for k in range(m.Nocc)])
        print(f"  mixing delta={dlt}: levels {np.round(e[:m.Nocc],3)} weights {np.array2string(wt, precision=3)} ||[H,rho]||={np.linalg.norm(H@o['rho']-o['rho']@H):.1e}")
