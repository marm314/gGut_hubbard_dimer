"""Ghosts on o and v: seed (B_c,B_o,B_v)=(3,3,3) from the cached B_c=3 solutions (ghost_solutions.npz) by adding to the o and v fragments
one 'sleeping ghost' pair each (bath orbital filled <-> qp orbital empty, bath empty <-> qp filled; as in bc5_cederbaum.embed),
perturb with noise and minimize. Same energy at the seed, so any energy gain / change of the core pole structure is due to the new ghosts.
python3 scripts/test_bo_bv_seeded.py V tc   -> log lines per run, cederbaum_data/bobv_seeded_V{V}_tc{tc}.npz (all runs)"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import exact_cederbaum as ed

V, tc = float(sys.argv[1]), float(sys.argv[2])
Z = np.load("cederbaum_data/ghost_solutions.npz")
z3 = Z[f"V{V}_tc{tc}_z3"]
m3 = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P3 = N.Problem(m3); st = P3.state(z3)
m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv", Bo=3, Bv=3); P = N.Problem(m)


def add_ghost_pair(phi, f_old, f_new):
    """phi in the sector of a fragment with B bath orbitals -> sector of the fragment with B+2 (the two new bath orbitals: first filled, second empty)."""
    B = f_old.B
    full = np.zeros(2 ** (2 * (1 + B))); full[f_old.sec] = phi
    ghost = np.zeros(16); ghost[12] = 1.0
    out = np.kron(full, ghost)[f_new.sec]
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12
    return out


xs = [st["phis"][0], add_ghost_pair(st["phis"][1], m3.frag[1], m.frag[1]), add_ghost_pair(st["phis"][2], m3.frag[2], m.frag[2])]
X = np.zeros((9, 5))
X[[0, 1, 2, 3, 6], :3] = st["X"]                  # old rows [c1 c2 c3 o v] -> new rows [c1 c2 c3 o1 v1]
X[5, 3] = 1.0; X[8, 4] = 1.0                      # o3, v3 filled; o2, v2 empty
z0 = P.pack(xs, X)
r = ed.solve(1.0, 10.0, V, tc, ht="cv")
e0, c0 = P.energy_and_grad(z0)[0], np.abs(P.constraints(z0)).max()
print(f"V={V:.4f} tc={tc}: ED E0={r['E0']:.9f}; seed E-E_ED={e0 - r['E0']:+.3e} cons={c0:.1e}", flush=True)
res_all = []
for noise in (1e-3, 3e-2, 1e-1):
    for sd in range(5):
        rng = np.random.default_rng(1000 * sd + int(noise * 1e4))
        res = N.minimize_once(P, z0 + noise * rng.standard_normal(z0.size), maxiter=600)
        c = np.abs(P.constraints(res.x)).max(); out = N.recover(P, res.x)
        print(f"noise={noise:.0e} seed={sd}: E-E_ED={res.fun - r['E0']:+.3e} (vs seed {res.fun - e0:+.1e}) cons={c:.0e} kkt={out['kkt']:.0e} comm={out['comm']:.0e} "
              f"nu_c={out['nu']:.6f} Rc2={out['R'][0] @ out['R'][0]:.4f} Ro={np.round(out['R'][1], 3)} Rv={np.round(out['R'][2], 3)} poles {np.round(out['poles'], 3)} w {np.array2string(out['weights'], precision=1)}", flush=True)
        res_all.append(dict(noise=noise, sd=sd, E=res.fun, cons=c, kkt=out["kkt"], comm=out["comm"], z=res.x, eps=out["eps"], poles=out["poles"], w=out["weights"], Hqp=out["Hqp"],
                            R0=out["R"][0], Ro=out["R"][1], Rv=out["R"][2]))
        np.save(f"cederbaum_data/bobv_seeded_V{V:.4f}_tc{tc}.npy", np.array(res_all, dtype=object), allow_pickle=True)
print("done", flush=True)
