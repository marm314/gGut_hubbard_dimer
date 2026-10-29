"""Search 1: other stationary solutions with core-ghost entanglement in Phi_c (B_c=3), i.e. a Delta_c with partially occupied ghost natural orbitals,
so that R_c can have weight on several qp levels (satellites). The solution found so far has natural occupations ~1e-7, 0.9999, 1 (core decoupled
from the ghosts). Seeds: Phi_c = the k-th eigenstate (k = 0..K-1, energy order) of a random embedding Hamiltonian H_imp^c with STRONG core-ghost
hybridization (scale s) and random bath mixing; Phi_o, Phi_v, X from the cached B_c=3 solution (ghost_solutions.npz). Each seed is minimized
(N.minimize_once, density constraints, analytic gradients) and classified: a 'new' solution is converged (|cons| < 1e-8, KKT_phi < 1e-5, iteration
limit not reached), has a Delta_c natural occupation in (0.01, 0.99) and weight off the main hole pole > 1e-3 (weights from the minimum-norm multipliers, N.recover) and |cons| < 1e-8.
python3 search_entangled_phi_c.py V tc [nseeds] [scales(comma list)] [K] [maxiter]  -> cederbaum_data/search_entangled_V{V}_tc{tc}.log / .npz (every run is logged and saved)"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import exact_cederbaum as ed

V, tc = float(sys.argv[1]), float(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
SCALES = [float(a) for a in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.5, 1.0, 2.0]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 4
MAXIT = int(sys.argv[6]) if len(sys.argv) > 6 else 4000
m = g.Model(1.0, 10.0, V, tc, 3, 1e-9, "cv"); P = N.Problem(m)
r = ed.solve(1.0, 10.0, V, tc, ht="cv")
Z = np.load("cederbaum_data/ghost_solutions.npz")[f"V{V}_tc{tc}_z3"]
st0 = P.state(Z)
LOG = open(f"cederbaum_data/search_entangled_V{V}_tc{tc}.log", "w")


def say(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def seed(rng, s, k):
    f = m.frag[0]; B = f.B
    hyb = rng.uniform(-1, 1, B) * s
    A = rng.standard_normal((B, B)); A = 0.5 * (A + A.T)
    lc = A * s * 2.0 + np.diag(rng.uniform(-6, 6, B))
    H = f.Hloc + sum(hyb[a] * f.hyb_op[a] for a in range(B)) + sum(lc[a, b] * f.dd[a][b] for a in range(B) for b in range(B))
    _, U = np.linalg.eigh(H)
    xs = [U[:, k] + 1e-3 * rng.standard_normal(U.shape[0]), st0["xs"][1], st0["xs"][2]]
    return P.pack(xs, st0["X"])


say(f"V={V:.4f} tc={tc}: ED E0={r['E0']:.9f}; cached solution E-E_ED={P.energy_and_grad(Z)[0] - r['E0']:+.2e}, natural occupations of Delta_c {np.round(np.linalg.eigvalsh(st0['Dl'][0]), 6)}")
rng = np.random.default_rng(0); runs = []; nnew = 0
for s in SCALES:
    for k in range(K):
        for sd in range(NS):
            t0 = time.time()
            try:
                res = N.minimize_once(P, seed(rng, s, k), maxiter=MAXIT)
                cons = np.abs(P.constraints(res.x)).max()
                o = N.recover(P, res.x)
                occ = np.linalg.eigvalsh(o["Dl"][0]); w = o["weights"]; woff = w.sum() - w.max()
                new = bool(cons < 1e-8 and o["kkt"] < 1e-5 and res.nit < MAXIT and np.any((occ > 0.01) & (occ < 0.99)) and woff > 1e-3)
                nnew += new
                say(f"s={s:g} k={k} seed{sd}: E-E_ED={res.fun - r['E0']:+.3e} cons={cons:.1e} it={res.nit} nu_c={o['nu']:.4f} occ={np.round(occ, 4)} "
                    f"poles={np.round(o['poles'], 2)} w={np.round(w, 4)} w_off={woff:.1e} KKT={o['kkt']:.1e} {'NEW' if new else ''} ({time.time() - t0:.0f} s)")
                runs.append(dict(s=s, k=k, sd=sd, x=res.x, E=res.fun - r["E0"], cons=cons, occ=occ, poles=o["poles"], w=w))
            except Exception as e:                                   # a failed run is logged, not hidden
                say(f"s={s:g} k={k} seed{sd}: FAILED {type(e).__name__}: {e}")
            np.savez(f"cederbaum_data/search_entangled_V{V}_tc{tc}.npz", **{f"run{i}_{key}": v for i, d in enumerate(runs) for key, v in d.items()})
say(f"done: {len(runs)} runs, {nnew} with entangled Delta_c and satellite weight")
