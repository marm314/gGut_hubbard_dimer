"""Impurity loop (gga_cederbaum.run, ground-state selection, follow=False) from random seeds.
Question: does the loop reach the direct-minimization point, or any other fixed point, and at what energy?
Usage: python3 loop_random_seeds.py V tc shift nseeds   (B_c=3, Delta=1, E_c=10)  -> one logged line per seed."""
import sys, time
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N

V, tc, shift, NSEED = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
Bc, NIT, MIX = 3, 300, 0.5
m = g.Model(1.0, 10.0, V, tc, Bc, shift, "cv"); P = N.Problem(m); rng0 = np.random.default_rng(0)
best = None
for k in range(5):
    res = N.minimize_once(P, N.random_start(P, rng0))
    if np.abs(P.constraints(res.x)).max() < 1e-9 and (best is None or res.fun < best.fun):
        best = res
Ed = best.fun
print(f"direct minimum: E={Ed:.10f}", flush=True)

Rg, lg = g.default_guess(m)
for seed in range(NSEED):
    rng = np.random.default_rng(100 + seed)
    kind = "perturbed" if seed % 2 == 0 else "random"
    if kind == "perturbed":
        R0 = [r * (1 + 0.5 * rng.standard_normal(r.shape)) for r in Rg]
        lam0 = g.noisy(lg, 1.0, rng)
    else:
        R0 = [rng.uniform(0.05, 1.0, b) for b in m.B]
        lam0 = []
        for b in m.B:
            e = rng.uniform(-12.0, 1.0, b)
            lam0.append(-np.diag(np.sort(e)))
        lam0[1] = np.array([[1.0 - 0.5]]); lam0[2] = np.array([[-1.0 + 0.5]])
    t0 = time.time()
    try:
        s = g.run(1.0, 10.0, V, tc, Bc, R0=R0, lam0=lam0, max_iter=NIT, mix=MIX, tol=1e-10, shift=shift, follow=False)
    except Exception as ex:
        print(f"seed {seed:2d} {kind:9s}: FAILED {type(ex).__name__}: {ex}", flush=True); continue
    pl, wt = s["poles"], s["weights"]
    print(f"seed {seed:2d} {kind:9s}: conv={s['converged']!s:5} it={s['iters']:3d} E={s['E_var']:.8f} "
          f"(E-E_direct={s['E_var']-Ed:+.2e}) resid={s['resid']:.1e} nu={s['nu']:.6f} Rc2={s['R'][0]@s['R'][0]:.4f} "
          f"main pole {pl[np.argmax(wt)]:.3f} (w={wt.max():.4f}) [{time.time()-t0:.0f}s]", flush=True)
