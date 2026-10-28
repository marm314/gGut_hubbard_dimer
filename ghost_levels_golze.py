"""Weightless ghost poles of the hole Green's function (only those that come from the ghosts) for B_c=3 and B_c=5, t_c=0.05.
A level of H_qp is a *ghost pole* if it is occupied in the qp density matrix rho (n_k = phi_k^T rho phi_k ~ 1, so it is a pole of
G^h), lives on the core+ghost block (>0.99) and has core-physical weight < 1e-4. (The occupied valence level o and the QP pole are
excluded; at B_c=1 there is no ghost and thus no ghost pole.) The constraint Jacobian leaves undetermined multiplier directions
(rank 7/8 at B_c=3, 13/17 at B_c=5); shifting the multipliers along them moves ghost levels without changing E, the main pole,
the weights or ||[H_qp,rho]||. Here the free ghost levels are placed at the exact satellites (first, second) measured from the gGut QP pole,
in units of Omega. Output cederbaum_data/golze_ghost_levels.npz. python3 ghost_levels_golze.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.optimize import least_squares
import gga_cederbaum as g
import exact_cederbaum as ed
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe
from bc5_cederbaum import embed
from compute_bc5_data import accepted

OM = 2.0
TCS = (0.1, 0.05)


def direct(V, Bc, seed):
    m = g.Model(1.0, 10.0, V, TC, Bc, 1e-9, "cv"); P = N.Problem(m); r = ed.solve(1.0, 10.0, V, TC, ht="cv"); rng = np.random.default_rng(seed); best = None
    for k in range(60):
        res = N.minimize_once(P, N.random_start(P, rng), maxiter=400)
        if accepted(P, res) is not None and (best is None or res.fun < best.fun): best = res
        if best is not None and best.fun - r["E0"] < 1.2e-7: break
    return m, P, best, r


out = {}
for TC in TCS:
  for V in pe.VS:
      r = ed.solve(1.0, 10.0, V, TC, ht="cv"); b = r["weights"] > 1e-3; pm = r["poles"][b]
      sat_rel = [(pm[1] - pm[0]) / OM, (pm[3] - pm[0]) / OM]        # first and second exact satellite from the exact QP, in Omega
      m3, P3, b3, _ = direct(V, 3, int(1000 * V) + 7)
      P5, z0 = embed(b3.x, V, TC); b5 = None
      for nz in (0.03, 0.1):
          for k in range(4):
              res = N.minimize_once(P5, z0 + nz * np.random.default_rng(31 * k + int(1000 * nz)).standard_normal(z0.size), maxiter=400)
              if accepted(P5, res) is not None and (b5 is None or res.fun < b5.fun): b5 = res
      for Bc, m, P, res in ((3, m3, P3, b3), (5, P5.m, P5, b5)):
          o = N.recover(P, res.x)
          J = P.cons_jac(res.x)[:, :P.nx]; U, s, Vt = np.linalg.svd(J, full_matrices=True)
          rank = int((s > 1e-6 * s[0]).sum()); null = U[:, rank:]
          E_, gz, st_, (hyb, ls, Tb, u, w) = P.energy_and_grad(res.x)
          mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)

          def analyse(mu):
              lam = [np.zeros((bb, bb)) for bb in m.B]; row = 0
              for I in range(3):
                  for a, bb in zip(*P.iu[I]):
                      lam[I][a, bb] = lam[I][bb, a] = (-mu[row] / 2.0 if a == bb else -mu[row] / 4.0); row += 1
              H = m.build_Hqp(o["R"], lam, Tb, u, w, o["nu"]); e, phi = np.linalg.eigh(H)
              occ = np.einsum("ik,ij,jk->k", phi, o["rho"], phi)
              blk0 = (phi[m.sl[0], :] ** 2).sum(0); wcore = (o["R"][0] @ phi[m.sl[0], :]) ** 2
              isg = (occ > 0.5) & (blk0 > 0.99) & (wcore < 1e-4)
              return e, occ, wcore, isg, np.linalg.norm(H @ o["rho"] - o["rho"] @ H)
          e, occ, wcore, isg, c0 = analyse(mu0)
          qp = e[np.argmax(wcore)]
          gh0 = np.sort(e[isg]); ng = len(gh0)
          print(f"tc={TC} V={V:.3f} B_c={Bc}: rank {rank}/{J.shape[0]} (null {null.shape[1]}); QP {qp:.3f}; ghost poles (occupied, weightless) at {np.round((gh0-qp)/OM,3)} Omega; comm {c0:.1e}", flush=True)
          # place the free ghost levels at the exact satellites
          tg = np.sort(np.array([qp + sat_rel[k] * OM for k in range(ng)]))
          def resid(a):
              e_, occ_, wc_, isg_, _ = analyse(mu0 + null @ a)
              gl = np.sort(e_[isg_])
              if len(gl) != ng: return np.full(ng, 1e3)
              return gl - tg
          # Gauss-Newton with a finite-difference Jacobian and step halving (the ghost levels move by ~0.1 per unit of the multiplier shift)
          na = null.shape[1]; a = np.zeros(na); f = resid(a); nf = np.linalg.norm(f)
          for it in range(200):
              if nf < 1e-7: break
              Jf = np.column_stack([(resid(a + 0.5 * np.eye(na)[k]) - f) / 0.5 for k in range(na)])
              step = -np.linalg.pinv(Jf, rcond=1e-6) @ f
              t = 1.0; ok = False
              while t > 1e-4:
                  fn = resid(a + t * step); nn = np.linalg.norm(fn)
                  if nn < nf: a, f, nf, ok = a + t * step, fn, nn, True; break
                  t *= 0.5
              if not ok: break
          best = (nf, a)
          a = best[1]; e2, occ2, wc2, isg2, c2 = analyse(mu0 + null @ a)
          gh = np.sort(e2[isg2]); wq = wc2[np.argmax(wc2)]
          print(f"   placed (residual {best[0]:.1e}): ghost poles at {np.round((gh-qp)/OM,3)} Omega; their weights {np.array2string(wc2[isg2][np.argsort(e2[isg2])],precision=1)}; QP weight {wq:.4f} (was {wcore[np.argmax(wcore)]:.4f}); comm {c2:.1e}", flush=True)
          out[f"V{V}_tc{TC}_B{Bc}_ghost_rel"] = (gh - qp) / OM; out[f"V{V}_tc{TC}_B{Bc}_ghost_rel_orig"] = (gh0 - qp) / OM
          out[f"V{V}_tc{TC}_B{Bc}_ghost_w"] = wc2[isg2][np.argsort(e2[isg2])]
np.savez("cederbaum_data/golze_ghost_levels.npz", **out)
print("done")
