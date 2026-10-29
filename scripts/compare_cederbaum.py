"""Step 4: gGut (c-v hopping) vs ED hole poles / weights of G_cc,
Delta=1, E_c=10, V in {0.5,1,2}, t_c in {0.5,0.2,0.1,0.05}; each t_c is seeded from the previous
converged/last iterate (R, lambda) with small random noise added to lambda. B_c in {1,3,5}."""
import sys
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed

Delta, Ec = 1.0, 10.0
TCS = [0.5, 0.2, 0.1, 0.05]
NOISE = 1e-3


def show(s, r, Bc):
    big = r["weights"] > 1e-3
    print(f"      ED  poles {np.round(r['poles'][big], 4)}  w {np.round(r['weights'][big], 4)}")
    keep = s["weights"] > 1e-6
    print(f"      gGut poles {np.round(s['poles'][keep], 4)}  w {np.round(s['weights'][keep], 4)}"
          f"   (all {len(s['poles'])} occupied qp levels: {np.round(s['poles'], 2)})")


if __name__ == "__main__":
    Bcs = [int(x) for x in sys.argv[1:]] or [1, 3, 5]
    rng = np.random.default_rng(11)
    for Bc in Bcs:
        for V in (0.5, 1.0, 2.0):
            print(f"\n=== B_c={Bc}  V={V} ===")
            seed = None
            for tc in TCS:
                r = ed.solve(Delta, Ec, V, tc, ht="cv")
                R0 = lam0 = None
                if seed is not None:
                    R0 = seed["R"]
                    lam0 = g.noisy(seed["lam"], NOISE, rng)
                s = g.run(Delta, Ec, V, tc, Bc, R0=R0, lam0=lam0, max_iter=800, mix=0.5, tol=1e-9)
                seed = s
                print(f"  t_c={tc}: conv={s['converged']} it={s['iters']}  E_gGut={s['E_var']:.8f}  E_ED={r['E0']:.8f}  "
                      f"dE={s['E_var']-r['E0']:+.2e}  nu_c={s['nu']:.6f} (ED {r['nu_c']:.6f})  resid={s['resid']:.1e}  Rc^T Rc={s['R'][0]@s['R'][0]:.4f}")
                show(s, r, Bc)
