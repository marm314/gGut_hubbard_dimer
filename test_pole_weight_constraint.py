"""Test 2 (gauge-invariant): put weight on a non-main hole pole of G_cc and see what it costs in energy (B_c=3, doc/cederbaum.tex
'Why there is no weight on the ghosts'). Replaces the frozen-direction constraint of test_rc_ghost_weight.py, which was not gauge invariant.
Constraint: w_off(z) = sum_n w_n - max_n w_n = f_target (imposed on sqrt(w_off), smooth at 0), with w_n = |R_c^T phi_n^(c)|^2 the weights of the occupied levels of H_qp, where
H_qp is built from the minimum-norm multipliers of the density constraints (N.recover; same prescription as in the figures).
The energy is minimized under the density constraints and this one (SLSQP, finite-difference Jacobian of the extra constraint), for a ladder
of f_target, starting from the cached B_c=3 solution plus noise, continuation along the ladder. The exact satellites carry 1 - w_QP.
Note: for f > 0 the point is not a stationary point of E alone (the extra constraint has its own multiplier), so this gives the energy price
E(f) - E_ED of satellite weight under that prescription, not a gGut solution.
python3 test_pole_weight_constraint.py V tc [f1 f2 ...]  -> cederbaum_data/pw_V{V}_tc{tc}.log / .npz"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import exact_cederbaum as ed

V, tc = float(sys.argv[1]), float(sys.argv[2])
FS = [float(a) for a in sys.argv[3:]] or [0.002, 0.01, 0.05, 0.1, 0.17]
m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
r = ed.solve(1.0, 10.0, V, tc, ht="cv")
LOG = open(f"cederbaum_data/pw_V{V}_tc{tc}.log", "w")


def say(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def w_off(z):
    w = N.recover(P, z)["weights"]
    return w.sum() - w.max()


def run(z0, f, maxiter=600):
    cons = [{"type": "eq", "fun": P.constraints, "jac": P.cons_jac}, {"type": "eq", "fun": lambda z: np.sqrt(max(w_off(z), 0.0)) - np.sqrt(f)}]
    return minimize(lambda z: P.energy_and_grad(z)[0], z0, jac=lambda z: P.energy_and_grad(z)[1],
                    constraints=cons, method="SLSQP", options=dict(maxiter=maxiter, ftol=1e-13))


z = np.load("cederbaum_data/ghost_solutions.npz")[f"V{V}_tc{tc}_z3"]
wqp = r["weights"].max()
t0 = time.time(); w0 = w_off(z)
say(f"V={V:.4f} tc={tc}: ED E0={r['E0']:.9f}; exact QP weight {wqp:.4f}, satellite weight {1 - wqp:.4f}; seed w_off={w0:.2e} ({time.time() - t0:.2f} s per evaluation)")
rng = np.random.default_rng(0); out = {}
for f in FS:
    best = None
    for noise in (1e-3, 1e-2, 3e-2):
        t0 = time.time()
        res = run(z + noise * rng.standard_normal(z.size), f)
        cc = max(np.abs(P.constraints(res.x)).max(), abs(np.sqrt(max(w_off(res.x), 0.0)) - np.sqrt(f)))
        say(f"  f_target={f:.3f} noise={noise:g}: E-E_ED={res.fun - r['E0']:+.3e} cons={cc:.1e} it={res.nit} ({res.message}) {time.time() - t0:.0f} s")
        if cc < 1e-7 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        say(f"f_target={f:.3f}: no feasible point found"); continue
    o = N.recover(P, best.x)
    say(f"f_target={f:.3f}: BEST E-E_ED={best.fun - r['E0']:+.3e} nu_c={o['nu']:.5f} |[Hqp,rho]|={o['comm']:.1e}\n"
        f"    poles {np.round(o['poles'], 3)} weights {np.round(o['weights'], 5)} (sum {o['weights'].sum():.4f})")
    k = f"f{f}"; out[k + "_x"], out[k + "_E"], out[k + "_poles"], out[k + "_w"] = best.x, best.fun - r["E0"], o["poles"], o["weights"]
    z = best.x
    np.savez(f"cederbaum_data/pw_V{V}_tc{tc}.npz", **out)
