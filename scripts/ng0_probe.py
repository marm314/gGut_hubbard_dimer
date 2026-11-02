import sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
import scan_ng2_not_imp_dV2_U as sc
t = 1.0; dV = float(sys.argv[1])
z = None
for U in np.round(np.arange(0.2, float(sys.argv[2]) + 1e-9, float(sys.argv[3])), 3):
    m = ni.NotImpDimer(0, U, t, deltaV=dV)
    if z is None:
        rng = np.random.default_rng(0); best = None
        for _ in range(20):
            r = m.minimize(m.random_start(rng))
            if np.abs(m.constraints(r.x)).max() < 1e-8 and (best is None or r.fun < best.fun): best = r
        z = best.x
    else: z = m.minimize(z).x
    a = m.analyse(z)
    print(f"U={U:6.2f} E={a['E_var']:.6f} Z0={a['Z0']:.5f} Z1={a['Z1']:.5f} eqp={np.round(a['eqp'],4)} |c|={np.abs(m.constraints(z)).max():.0e}")
