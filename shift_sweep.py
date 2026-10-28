"""Does the impurity loop (gga_cederbaum.run) stay at / converge near the direct-minimization solution?
Usage: python3 shift_sweep.py V [shifts...]. Sweep the regularization shift; for each shift: direct minimization (best of NS starts), then the loop started
from the full direct state (state0) with follow in {False, True} and mix in {1.0, 0.5, 0.2}. For converged loop
points the direct minimizer is restarted from the loop point to see whether the energy can still decrease."""
import os
import sys
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N

V, tc, Bc = float(sys.argv[1]), 0.5, 3
SHIFTS = [float(x) for x in sys.argv[2:]] or [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
NS, NIT = int(os.environ.get("NS", 5)), 300

for shift in SHIFTS:
    m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng = np.random.default_rng(0)
    best = None
    for k in range(NS):
        res = N.minimize_once(P, N.random_start(P, rng))
        if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
            best = res
    out = N.recover(P, best.x); st = out["st"]; q0 = st["q"][0]
    state0 = dict(R=out["R"], lam=out["lam"], nu=q0["nu"], K=q0["K"], rho=out["rho"], hyb=out["hyb"],
                  lamc=out["lamc"], Dl=out["Dl"])
    print(f"\n=== shift {shift:g}: direct E={best.fun:.8f} KKT={out['kkt']:.1e} eig(Delta_c)={np.linalg.eigvalsh(out['Dl'][0])}", flush=True)
    for follow in (False, True):
        for mix in (1.0, 0.5, 0.2):
            s = g.run(1.0, 10.0, V, tc, Bc, max_iter=NIT, mix=mix, tol=1e-10, follow=follow,
                      phi0=st["phis"], state0=state0, shift=shift)
            line = (f"  follow={follow!s:5} mix={mix}: conv={s['converged']!s:5} it={s['iters']:3d} E_loop={s['E_var']:.8f} "
                    f"(E_loop-E_direct={s['E_var']-best.fun:+.2e}) resid={s['resid']:.1e} nu={s['nu']:.6f} "
                    f"Rc2={s['R'][0]@s['R'][0]:.4f} main pole {s['poles'][np.argmax(s['weights'])]:.3f} (w={s['weights'].max():.4f})")
            if s["converged"]:
                xs = [i["gs"] for i in s["imp"]]
                _, Uq = np.linalg.eigh(s["Hqp"])
                z0 = P.pack(xs, Uq[:, :m.Nocc])
                r2 = N.minimize_once(P, z0)
                c2 = np.abs(P.constraints(r2.x)).max()
                line += f"  | direct restarted from loop point: E={r2.fun:.8f} (dE={r2.fun-s['E_var']:+.1e}, cons {c2:.0e})"
            print(line, flush=True)
