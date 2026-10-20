"""B_c=5 by direct minimization, made cheap by seeding from the B_c=3 solution.

Embedding of a B_c=3 solution into B_c=5: the two new core bath orbitals form a pure 'sleeping ghost' pair
(bath orbital 3 filled <-> qp orbital c4 empty, bath orbital 4 empty <-> qp orbital c5 filled), so
Phi_5 = Phi_3 (x) |ghost>, rho_5 = rho_3 (+) diag(0,1) in the core ghost block, and the constraints
rho^II = Delta^II(Phi_I) hold exactly with the same energy. Random perturbations of this feasible point break the
degeneracy and let the minimizer decide whether the extra ghosts lower the energy.
Independent (V, t_c, seed) tasks run in parallel (one BLAS thread each)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import numpy as np
from multiprocessing import Pool

import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N

CASES = [(0.5, 0.1), (1.0, 0.1), (2.0, 0.1), (0.5, 0.05), (1.0, 0.05), (2.0, 0.05)]
NOISES = [1e-3, 3e-2]
NSEEDS = 6


def solve_bc3(args):
    V, tc = args
    m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m); rng = np.random.default_rng(100 + int(1000 * V + 7 * tc * 100))
    best = None
    for k in range(10):
        res = N.minimize_once(P, N.random_start(P, rng))
        if np.abs(P.constraints(res.x)).max() < 1e-9:
            o = N.recover(P, res.x)
            if o["kkt"] < 1e-5 and (best is None or res.fun < best.fun):
                best = res
            if best is not None and best.fun - ed.solve(1.0, 10.0, V, tc, ht="cv")["E0"] < 2e-7:
                break
    return (V, tc, best.x if best is not None else None)


def embed(z3, V, tc):
    """B_c=3 solution z3 -> feasible B_c=5 point (same energy)."""
    m3 = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P3 = N.Problem(m3)
    m5 = g.Model(1.0, 10.0, V, tc, 5, 1e-9, "cv"); P5 = N.Problem(m5)
    st = P3.state(z3)
    f3, f5 = m3.frag[0], m5.frag[0]
    full3 = np.zeros(f3.Hloc.shape[0] * 0 + 2 ** 8); full3[f3.sec] = st["phis"][0]
    ghost = np.zeros(16); ghost[12] = 1.0                       # n8=n9=1 (bath 3 filled), n10=n11=0
    full5 = np.kron(full3, ghost)
    phi5 = full5[f5.sec]
    assert abs(np.linalg.norm(phi5) - 1.0) < 1e-12
    X3 = st["X"]                                                # rows [c1 c2 c3 o v] x 3 occupied columns
    X5 = np.zeros((7, 4))
    rows = [0, 1, 2, 5, 6]
    X5[rows, :3] = X3
    X5[4, 3] = 1.0                                              # c5 filled
    xs = [phi5, st["phis"][1], st["phis"][2]]
    return P5, P5.pack(xs, X5)


def run_bc5(args):
    V, tc, z3, noise, seed = args
    P5, z0 = embed(z3, V, tc)
    rng = np.random.default_rng(seed)
    e0 = P5.energy_and_grad(z0)[0]; c0 = np.abs(P5.constraints(z0)).max()
    z = z0 + noise * rng.standard_normal(z0.size)
    res = N.minimize_once(P5, z, maxiter=500)
    cons = np.abs(P5.constraints(res.x)).max()
    out = N.recover(P5, res.x)
    return dict(V=V, tc=tc, noise=noise, seed=seed, E=res.fun, E_seed=e0, cons0=c0, cons=cons, kkt=out["kkt"], comm=out["comm"],
                nu=out["nu"], poles=out["poles"], w=out["weights"], Rc2=float(out["R"][0] @ out["R"][0]), nit=res.nit,
                eig_D=np.linalg.eigvalsh(out["Dl"][0]))


if __name__ == "__main__":
    ncpu = int(os.environ.get("NPROC", "12"))
    with Pool(ncpu) as pool:
        sols = pool.map(solve_bc3, CASES)
        print("B_c=3 seeds:", [(V, tc, s is not None) for V, tc, s in sols], flush=True)
        tasks = [(V, tc, z3, nz, 1000 * k + int(100 * V) + int(1000 * tc)) for (V, tc, z3) in sols if z3 is not None
                 for nz in NOISES for k in range(NSEEDS)]
        results = pool.map(run_bc5, tasks, chunksize=1)
    ref = {(V, tc): ed.solve(1.0, 10.0, V, tc, ht="cv") for V, tc in CASES}
    np.save("bc5_results.npy", np.array(results, dtype=object), allow_pickle=True)
    for V, tc in CASES:
        r = ref[(V, tc)]
        rs = sorted([x for x in results if x["V"] == V and x["tc"] == tc], key=lambda x: x["E"])
        print(f"\n=== V={V} t_c={tc}: ED E0={r['E0']:.8f}; B_c=3 seed E={rs[0]['E_seed']:.8f} (dE={rs[0]['E_seed']-r['E0']:+.2e})")
        for x in rs[:6]:
            print(f"  noise={x['noise']:.0e} seed={x['seed']:5d}: E-E_ED={x['E']-r['E0']:+.3e} (vs seed {x['E']-x['E_seed']:+.2e}) cons={x['cons']:.0e} "
                  f"KKT={x['kkt']:.0e} nu={x['nu']:.6f} Rc2={x['Rc2']:.4f} it={x['nit']} poles={np.round(x['poles'],3)} w={np.round(x['w'],4)}", flush=True)
