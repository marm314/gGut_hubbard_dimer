"""Redo the B_c=3 points of cederbaum_energies.npz whose direct minimization ended in a poor local minimum
(E-E_ED > 2e-7 or KKT > 1e-4): up to NTRY random starts, keep the lowest feasible point that is also well converged."""
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
from plot_cederbaum_energies import VS, TCS, FILE

d = dict(np.load(FILE))
rng = np.random.default_rng(77)
NTRY = 14
for V in VS:
    for tc in TCS:
        key = f"V{V}_tc{tc}"
        row = d[key]
        if row[4] - row[0] <= 2e-7 and row[8] <= 1e-4:
            continue
        m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
        best = None
        for k in range(NTRY):
            res = N.minimize_once(P, N.random_start(P, rng))
            if np.abs(P.constraints(res.x)).max() > 1e-9:
                continue
            o3 = N.recover(P, res.x)
            good = (res.fun - row[0] < 2e-7) and o3["kkt"] < 1e-4
            if best is None or res.fun < best[0].fun:
                best = (res, o3)
            if good:
                best = (res, o3); break
        if best is not None:
            res, o3 = best
            d[key] = np.array([row[0], row[1], row[2], row[3], res.fun, o3["nu"], row[6], np.abs(P.constraints(res.x)).max(), o3["kkt"]])
            print(key, "old dE %.2e kkt %.0e -> new dE %.2e kkt %.0e (tries %d)" % (row[4] - row[0], row[8], res.fun - row[0], o3["kkt"], k + 1), flush=True)
        else:
            print(key, "no feasible point found", flush=True)
np.savez(FILE, **d)
print("saved")
