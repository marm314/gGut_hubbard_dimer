"""Test: can R_c have weight on a ghost? (why gGut has no satellites, B_c=3, doc/cederbaum.tex 'Why there is no weight on the ghosts')
The hole weights are |R_c^T phi_n|^2, so a satellite needs R_c to leave the dominant natural orbital of ^c Delta (the nearly full core
orbital) and enter the ghost natural orbitals. Here the energy is minimized with the extra equality constraint
    f(z)^2 = 1 - (R_c . w_max)^2 / (R_c . R_c) = f_target with the frozen w_max,     w_max = natural orbital of ^c Delta along which R_c is largest (the core orbital),
for a ladder of f_target (continuation from the previous point, noisy restarts), starting from the cached B_c=3 solution (ghost_solutions.npz).
Result: the energy cost E(f)-E_ED of putting a fraction f of R_c on the ghosts (f = 0: the solution found before) and the resulting
hole poles/weights (multipliers = minimum-norm ones, so poles of zero weight are arbitrary, see the text). The exact satellites carry
1 - w_QP of the core weight (printed).
python3 test_rc_ghost_weight.py V tc [f1 f2 ...]   -> log lines, cederbaum_data/rc_ghost_V{V}_tc{tc}.npz (all points), cederbaum_data/rc_ghost_V{V}_tc{tc}.log"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.optimize import minimize
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import exact_cederbaum as ed

V, tc = float(sys.argv[1]), float(sys.argv[2])
FS = [float(a) for a in sys.argv[3:]] or [0.0, 0.01, 0.02, 0.05, 0.1, 0.17]
m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
r = ed.solve(1.0, 10.0, V, tc, ht="cv")
LOG = open(f"cederbaum_data/rc_ghost_V{V}_tc{tc}.log", "w")


def say(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


from gga_dimer import _sqrt_eig


def frac_and_grad(z, w0):
    """f = 1 - (R_c.w0)^2/(R_c.R_c) with w0 the fixed core direction (frozen from the seed; the natural orbitals of ^c Delta have occupations 1 and
    0.9999, so their eigenvectors are ill-conditioned), R_c = S(Delta_c)^-1 D_c, and its gradient with respect to z (only phi_c enters).
    dR = S^-1 dD - S^-1 dS R, dS_ij = dDelta_ij (1-d_i-d_j)/(s_i+s_j) in the eigenbasis of Delta_c, dDelta = -dDbb  (cf. gga_dimer.grad_S)."""
    st = P.state(z); R = st["R"][0]; Dl = st["Dl"][0]; B = m.B[0]
    RR, Rw = R @ R, R @ w0
    f = 1.0 - Rw * Rw / RR
    rbar = -2.0 * Rw * w0 / RR + 2.0 * Rw * Rw * R / RR ** 2                     # df/dR
    d, W, sv = _sqrt_eig(Dl, m.shift)
    q = W @ ((W.T @ rbar) / sv)                                                  # S^-1 rbar
    w = (1.0 - d[:, None] - d[None, :]) / (sv[:, None] + sv[None, :])
    M = W @ (np.outer(W.T @ q, W.T @ R) * w) @ W.T                               # df = q.dD + sum M_ab dDbb_ab
    phi = st["phis"][0]; dim = phi.size; S = P.sp[0]; T = st["q"][0]["T"]
    Tdc = (S["DC"] @ phi).reshape(B, dim); Tdct = (S["DCT"] @ phi).reshape(B, dim)
    g = 0.5 * (q @ (Tdc + Tdct)) + 0.5 * ((M + M.T).reshape(1, B * B) @ T).ravel()
    nrm = np.linalg.norm(st["xs"][0]); g = (g - phi * (phi @ g)) / nrm
    gz = np.zeros(z.size); gz[:dim] = g
    return f, gz


def frac(z):
    return frac_and_grad(z, W0)[0]


def run(z0, f, maxiter=400):
    cons = [{"type": "eq", "fun": P.constraints, "jac": P.cons_jac},
            {"type": "eq", "fun": lambda z: frac_and_grad(z, W0)[0] - f, "jac": lambda z: frac_and_grad(z, W0)[1][None, :]}]
    return minimize(lambda z: P.energy_and_grad(z)[0], z0, jac=lambda z: P.energy_and_grad(z)[1],
                    constraints=cons, method="SLSQP", options=dict(maxiter=maxiter, ftol=1e-13))


z = np.load("cederbaum_data/ghost_solutions.npz")[f"V{V}_tc{tc}_z3"]
wqp = r["weights"].max()
_st = P.state(z); _n, _W = np.linalg.eigh(_st["Dl"][0]); W0 = _W[:, np.argmax(np.abs(_W.T @ _st["R"][0]))]      # frozen core direction
_zt = z + 1e-4 * np.random.default_rng(1).standard_normal(z.size); _v = np.random.default_rng(2).standard_normal(z.size)
_fd = (frac(_zt + 1e-6 * _v) - frac(_zt - 1e-6 * _v)) / 2e-6
say(f"gradient check of f: analytic {frac_and_grad(_zt, W0)[1] @ _v:.6e}  finite difference {_fd:.6e}")
say(f"V={V:.4f} tc={tc}: ED E0={r['E0']:.9f}; exact QP weight {wqp:.4f}, satellite weight {1 - wqp:.4f} (t_c=0 value in the text); seed f={frac(z) ** 2:.2e}")
rng = np.random.default_rng(0)
out = {}
for f in FS:
    best = None
    for noise in (1e-2, 3e-2, 1e-1):
        res = run(z + noise * rng.standard_normal(z.size), f)
        cc = max(np.abs(P.constraints(res.x)).max(), abs(frac(res.x) - f))
        say(f"  f_target={f:.3f} noise={noise:g}: E-E_ED={res.fun - r['E0']:+.3e} cons={cc:.1e} it={res.nit} ({res.message})")
        if cc < 1e-7 and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        say(f"f_target={f:.3f}: no feasible point found"); continue
    o = N.recover(P, best.x)
    say(f"f_target={f:.3f}: BEST E-E_ED={best.fun - r['E0']:+.3e} nu_c={o['nu']:.5f} KKT={o['kkt']:.1e} |[Hqp,rho]|={o['comm']:.1e}\n"
        f"    poles {np.round(o['poles'], 3)} weights {np.round(o['weights'], 5)} (sum {o['weights'].sum():.4f})")
    k = f"f{f}"
    out[k + "_x"], out[k + "_E"], out[k + "_poles"], out[k + "_w"] = best.x, best.fun - r["E0"], o["poles"], o["weights"]
    z = best.x                                                              # continuation
    np.savez(f"cederbaum_data/rc_ghost_V{V}_tc{tc}.npz", **out)
