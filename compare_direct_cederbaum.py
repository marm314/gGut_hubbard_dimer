"""Step 4 with the direct-minimization solver (not_imp_gga_cederbaum.py): gGut vs ED for the hole poles /
weights of G_cc. Delta=1, E_c=10, V in {0.5,1,2}, t_c in {0.5,0.2,0.1,0.05}. The first t_c takes the lowest
feasible minimum of NS random starts; each further t_c is seeded from the previous solution with small random noise
(NOISE) on all variables. B_c given on the command line."""
import sys
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N

Delta, Ec = 1.0, 10.0
TCS = [0.5, 0.2, 0.1, 0.05]
NOISE = 1e-3
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

if __name__ == "__main__":
    Bc = int(sys.argv[1])
    rng = np.random.default_rng(5)
    for V in (0.5, 1.0, 2.0):
        print(f"\n=== B_c={Bc} (direct minimization)  V={V} ===", flush=True)
        zprev = None
        for tc in TCS:
            m = g.Model(Delta, Ec, V, tc, Bc, 1e-9, "cv")
            P = N.Problem(m)
            r = ed.solve(Delta, Ec, V, tc, ht="cv")
            if zprev is None:
                cands = []
                for k in range(NS):
                    res = N.minimize_once(P, N.random_start(P, rng))
                    if np.abs(P.constraints(res.x)).max() < 1e-8:
                        cands.append(res)
                res = min(cands, key=lambda x: x.fun) if cands else None
            else:
                z0 = zprev + NOISE * rng.standard_normal(zprev.size)
                res = N.minimize_once(P, z0)
            if res is None:
                print(f"  t_c={tc}: no feasible minimum found", flush=True)
                continue
            zprev = res.x
            out = N.recover(P, res.x)
            cons = np.abs(P.constraints(res.x)).max()
            print(f"  t_c={tc}: E_gGut={res.fun:.8f} E_ED={r['E0']:.8f} dE={res.fun-r['E0']:+.2e} nu_c={out['nu']:.6f} "
                  f"(ED {r['nu_c']:.6f}) |cons|={cons:.1e} KKT_phi={out['kkt']:.1e} |[Hqp,rho]|={out['comm']:.1e} "
                  f"Rc^TRc={out['R'][0]@out['R'][0]:.4f}")
            big = r["weights"] > 1e-3
            print(f"      ED   poles {np.round(r['poles'][big],4)}  w {np.round(r['weights'][big],4)}")
            print(f"      gGut occupied qp levels {np.round(out['poles'],4)}  core weights {np.round(out['weights'],5)}", flush=True)
