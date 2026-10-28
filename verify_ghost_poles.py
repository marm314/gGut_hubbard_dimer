"""Verify the nature of every level of H_qp for the solutions behind the ghost-pole dots (t_c=0.1, 0.05; both V; B_c=3, 5):
energy, occupation n_k = phi_k^T rho phi_k in the actual qp density matrix, weight on the core+ghost block / o block / v block,
core-physical weight |R_c^T phi|^2. Same seeds as ghost_levels_golze.py. Compares the ghost poles found here with the *_orig data.
python3 verify_ghost_poles.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe
from bc5_cederbaum import embed
from compute_bc5_data import accepted

OM = 2.0
GL = np.load("cederbaum_data/golze_ghost_levels.npz")
CACHE = "cederbaum_data/ghost_solutions.npz"          # minimizer solutions z (B_c=3, 5) per (t_c, V): computed once, then loaded
cache = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}


def direct(V, TC, Bc, seed):
    m = g.Model(1.0, 10.0, V, TC, Bc, 1e-9, "cv"); P = N.Problem(m); r = ed.solve(1.0, 10.0, V, TC, ht="cv"); rng = np.random.default_rng(seed); best = None
    for k in range(60):
        res = N.minimize_once(P, N.random_start(P, rng), maxiter=400)
        if accepted(P, res) is not None and (best is None or res.fun < best.fun): best = res
        if best is not None and best.fun - r["E0"] < 1.2e-7: break
    return m, P, best


for TC in (0.1, 0.05):
    for V in pe.VS:
        k3, k5 = f"V{V}_tc{TC}_z3", f"V{V}_tc{TC}_z5"
        m3 = g.Model(1.0, 10.0, V, TC, 3, 1e-9, "cv"); P3 = N.Problem(m3)
        if k3 not in cache:
            _, _, b3 = direct(V, TC, 3, int(1000 * V) + 7); cache[k3] = b3.x
            P5, z0 = embed(b3.x, V, TC); b5 = None
            for nz in (0.03, 0.1):
                for k in range(4):
                    res = N.minimize_once(P5, z0 + nz * np.random.default_rng(31 * k + int(1000 * nz)).standard_normal(z0.size), maxiter=400)
                    if accepted(P5, res) is not None and (b5 is None or res.fun < b5.fun): b5 = res
            cache[k5] = b5.x; np.savez(CACHE, **cache)      # save as soon as a (t_c, V) is done
        P5, _ = embed(cache[k3], V, TC)
        for Bc, m, P, zsol in ((3, m3, P3, cache[k3]), (5, P5.m, P5, cache[k5])):
            class _R: pass
            res = _R(); res.x = zsol
            o = N.recover(P, res.x); e, phi = np.linalg.eigh(o["Hqp"])
            occ = np.einsum("ik,ij,jk->k", phi, o["rho"], phi)
            blk = np.array([(phi[m.sl[I], :] ** 2).sum(0) for I in range(3)]).T
            wc = (o["R"][0] @ phi[m.sl[0], :]) ** 2
            qp = e[np.argmax(wc)]
            print(f"\nt_c={TC} V/g={V:.3f} B_c={Bc}: H_qp has {m.n} levels = core block {m.B[0]} (core + {m.B[0]-1} ghost orbitals) + o + v; {m.Nocc} occupied in rho")
            print("   energy   occ(rho)  w_core+ghost  w_o   w_v   core-weight   -> character")
            ghosts = []
            for k in range(m.n):
                if wc[k] > 1e-3: ch = "QP (core) pole"
                elif blk[k, 0] > 0.99 and occ[k] > 0.5: ch = "GHOST, occupied -> zero-weight pole of G^h"; ghosts.append(e[k])
                elif blk[k, 0] > 0.99: ch = "ghost, EMPTY (particle side; not a pole of G^h)"
                elif blk[k, 1] > 0.99 and occ[k] > 0.5: ch = "valence o, occupied"
                elif blk[k, 1] > 0.99: ch = "valence o, empty"
                elif blk[k, 2] > 0.99: ch = "valence v, " + ("occupied" if occ[k] > 0.5 else "empty")
                else: ch = "mixed block"
                print(f"  {e[k]:8.3f}  {occ[k]:7.3f}   {blk[k,0]:8.4f}   {blk[k,1]:5.3f} {blk[k,2]:5.3f}   {wc[k]:9.1e}     {ch}")
            ghosts = (np.sort(ghosts) - qp) / OM
            ref = np.sort(GL[f"V{V}_tc{TC}_B{Bc}_ghost_rel_orig"])
            same = len(ghosts) == len(ref) and np.allclose(ghosts, ref, atol=1e-6)
            print(f"   ghost poles here (rel. QP, Omega): {np.round(ghosts,3)}; plotted (orig): {np.round(ref,3)}; identical: {same}", flush=True)
print("done")
