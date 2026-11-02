"""Ng=0 (ordinary Gutzwiller) dimer, dV=0: closed forms of gGut_notes.pdf Sec. III D against NotImpDimer(Ng=0): Z=1-(U/8t)^2, E=-2t-U/2-U^2/(32t), docc=1/4-U/(32t) (U<=8t); E=-U, Z=0, docc=0 above."""
import sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
t = 1.0
for U in (0.5, 2.0, 5.0, 7.5, 8.5, 10.0):
    m = ni.NotImpDimer(0, U, t, deltaV=0.0); rng = np.random.default_rng(0); best = None
    for _ in range(10):
        r = m.minimize(m.random_start(rng))
        if np.abs(m.constraints(r.x)).max() < 1e-8 and (best is None or r.fun < best.fun): best = r
    a = m.analyse(best.x)
    Za = max(1 - (U / 8) ** 2, 0.0); Ea = -2 - U / 2 - U**2 / 32 if U <= 8 else -U; da = 0.25 - U / 32 if U <= 8 else 0.0
    print(f"U={U:4.1f}  E num={a['E_var']:.8f} ana={Ea:.8f} | Z0 num={a['Z0']:.6f} ana={Za:.6f} | docc num={a['docc']:.6f} ana={da:.6f} | eqp={np.round(a['eqp'],4)}")
