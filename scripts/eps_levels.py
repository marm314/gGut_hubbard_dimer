"""Dependence of the H_qp levels (QP core, o, v), weights, energy and commutator on the regularizing shift eps at small t_c (B_c=3), by re-minimizing from the stored solutions.
PYTHONPATH=scripts python3 scripts/eps_levels.py"""
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g, not_imp_gga_cederbaum as N, exact_cederbaum as ed
Z9 = np.load("cederbaum_data/ghost_solutions.npz"); Z7 = np.load
for V in (0.7071067811865476, 1.4142135623730951):
    for tc in (0.05, 0.1):
        E0 = ed.solve(1.0, 10.0, V, tc, ht="cv")["E0"]; z7 = np.load(f"cederbaum_data/scan_ghost_tc_eps1e-07/V{V}.npz")[f"tc{tc}_B3_z"]
        for eps, z0 in ((1e-9, Z9[f"V{V}_tc{tc}_z3"]), (1e-8, z7), (1e-7, z7), (1e-6, z7)):
            m = g.Model(1.0, 10.0, V, tc, 3, eps, "cv"); P = N.Problem(m); z = z0.copy(); cons = {"type": "eq", "fun": P.constraints, "jac": P.cons_jac}
            if eps != 1e-9:
                for _ in range(3):
                    r = minimize(lambda x: P.energy_and_grad(x)[0], z, jac=lambda x: P.energy_and_grad(x)[1], constraints=[cons], method="SLSQP", options=dict(maxiter=1500, ftol=1e-15)); z = r.x
            o = N.recover(P, z); e, ph = np.linalg.eigh(o["Hqp"]); R = o["R"]; W = np.array([[(R[I] @ ph[m.sl[I], k]) ** 2 for k in range(5)] for I in range(3)])
            qp, io, iv = [int(np.argmax(W[I])) for I in range(3)]
            print(f"V/g={V:.3f} tc={tc} eps={eps:.0e}: E-E_ED {o['E']-E0:.2e}; QP core {e[qp]:.4f} (w {W[0,qp]:.4f}); o level {e[io]:.4f} (R_o^2 {R[1]@R[1]:.4f}); v level {e[iv]:.4f}; comm {o['comm']:.1e}; |cons| {np.abs(P.constraints(z)).max():.0e}", flush=True)
