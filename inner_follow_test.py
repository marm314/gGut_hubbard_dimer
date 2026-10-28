"""Inner loop on rho (gga_cederbaum_inner) started from the direct-minimization state, follow=True (MOM-like).
Usage: python3 inner_follow_test.py V shift [mix...]. Direct minimization (NS starts) once, then the loop with inner in {1, 50}."""
import os, sys, pickle, numpy as np
import gga_cederbaum_inner as g
import not_imp_gga_cederbaum as N

V, shift = float(sys.argv[1]), float(sys.argv[2])
MIXES = [float(x) for x in sys.argv[3:]] or [1.0, 0.5]
tc, Bc, NS, NIT = 0.5, 3, int(os.environ.get("NS", 5)), 300
m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng = np.random.default_rng(0)
best = None
for k in range(NS):
    res = N.minimize_once(P, N.random_start(P, rng))
    if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
        best = res
out = N.recover(P, best.x); st = out["st"]; q0 = st["q"][0]
state0 = dict(R=out["R"], lam=out["lam"], nu=q0["nu"], K=q0["K"], rho=out["rho"], hyb=out["hyb"], lamc=out["lamc"], Dl=out["Dl"])
print(f"V={V} shift={shift:g}: direct E={best.fun:.10f} KKT={out['kkt']:.1e}", flush=True)
res_all = {}
for mix in MIXES:
    for inner in (1, 50):
        s = g.run(1.0, 10.0, V, tc, Bc, max_iter=NIT, mix=mix, tol=1e-10, follow=True, phi0=st["phis"],
                  state0=state0, shift=shift, inner=inner)
        print(f"  mix={mix} inner={inner:2d}: conv={s['converged']!s:5} it={s['iters']:3d} E_loop-E_direct={s['E_var']-best.fun:+.2e} "
              f"resid={s['resid']:.1e} nu={s['nu']:.6f} main pole {s['poles'][np.argmax(s['weights'])]:.3f} (w={s['weights'].max():.4f})", flush=True)
        res_all[(mix, inner)] = dict(conv=s["converged"], it=s["iters"], E=s["E_var"], resid=s["resid"], hist=s["hist"], poles=s["poles"], weights=s["weights"])
pickle.dump(dict(E_direct=best.fun, state0=state0, results=res_all), open(f"cederbaum_data/inner_follow_V{V}_s{shift:g}.pkl", "wb"))
