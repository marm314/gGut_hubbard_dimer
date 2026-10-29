"""Compare the standard loop (inner=1) with an inner loop on rho (inner>1) for the t_c-Cederbaum model."""
import sys, time, numpy as np
import gga_cederbaum_inner as g
Bc, V, tc = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
for inner in (1, 50):
    t = time.time()
    s = g.run(1.0, 10.0, V, tc, Bc, max_iter=800, mix=0.5, tol=1e-9, inner=inner)
    print(f"Bc={Bc} V={V} tc={tc} inner={inner}: conv={s['converged']} it={s['iters']} E={s['E_var']:.10f} "
          f"nu_c={s['nu']:.6f} resid={s['resid']:.1e} poles={np.round(s['poles'],4)} w={np.round(s['weights'],4)} ({time.time()-t:.1f}s)", flush=True)
