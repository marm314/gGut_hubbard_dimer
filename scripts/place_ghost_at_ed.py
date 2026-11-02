"""Move the weightless ghost level(s) onto ED satellite(s) by the gauge shift (null directions of J_phi^T) and list what changes vs the min-norm gauge.
B_c=3: one ghost -> the ED satellite nearest -12. B_c=5: two ghosts -> first and second ED satellite (poles with w>1e-3 at index 1 and 3, as in ghost_levels_golze.py).
GHOST_EPS=1e-7 python3 scripts/place_ghost_at_ed.py V tc [Bc=3]"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
from scipy.optimize import least_squares
import gga_cederbaum as g, exact_cederbaum as exm, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = f"_eps{EPS:g}"; V = float(sys.argv[1]); tc = float(sys.argv[2]); Bc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
def himp(m, I, hyb, lamc, alpha, Xi):
    f = m.frag[I]; B = f.B; H = f.Hloc.copy()
    for a in range(B):
        H += hyb[a] * f.hyb_op[a]
        for b in range(B): H += lamc[a, b] * f.dd[a][b]
    if Xi is not None:
        H += (alpha + 2.0 * np.trace(Xi)) * f.nc
        for gg in range(B):
            for d in range(B): H -= Xi[gg, d] * f.nc_dd[gg][d]
    return H
data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz"); z = data[f"tc{tc}_B{Bc}_z"]
if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc, EPS); m = P.m
E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None); na = null.shape[1]
rho, R, Dl, q = st["rho"], st["R"], st["Dl"], st["q"]; alpha, Xi = m.core_extra(rho, R, Dl[0]); ng = (Bc - 1) // 2
def lam_of(a):
    mu = mu0 + null @ a; lam = [np.zeros((b, b)) for b in m.B]; row = 0
    for I in range(3):
        for i, j in zip(*P.iu[I]): lam[I][i, j] = lam[I][j, i] = (-mu[row] / 2.0 if i == j else -mu[row] / 4.0); row += 1
    return lam
def info(a):
    lam = lam_of(a); Hq = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, ph = np.linalg.eigh(Hq); occ = np.einsum("ik,ij,jk->k", ph, rho, ph); wc = (R[0] @ ph[m.sl[0], :]) ** 2
    blk = [(ph[m.sl[I], :] ** 2).sum(0) for I in range(3)]; lamc = [ls[I] - lam[I] for I in range(3)]
    Hi = himp(m, 0, hyb[0], lamc[0], alpha, Xi); ev, U = np.linalg.eigh(Hi); phi = st["phis"][0]; k = int(np.argmax((U.T @ phi) ** 2))
    isg = (occ > 0.5) & (blk[0] > 0.9) & (wc < 1e-2)
    return dict(lam=lam, lamc=lamc, e=e, ph=ph, occ=occ, wc=wc, blk=blk, Hq=Hq, ev=ev, k=k, resid=np.linalg.norm(Hi @ phi - (phi @ Hi @ phi) * phi), isg=isg)
rr = exm.solve(1.0, 10.0, V, tc, ht="cv"); sats = rr["poles"][rr["weights"] > 1e-3]; wsat = rr["weights"][rr["weights"] > 1e-3]
targets = np.array([sats[np.argmin(abs(sats + 12.0))]]) if Bc == 3 else np.sort(np.array([sats[1], sats[3]]))
# H_qp(a) = H_qp(0) + sum_j a_j D_j is EXACTLY linear in the gauge parameters a. Aim: shift the occupied-ghost levels by delta_k keeping their orbitals
# (T = sum_k delta_k |g_k><g_k|), everything else fixed; solve sum_j a_j D_j = T by linear least squares (residual shows whether it is reachable).
A0 = info(np.zeros(na)); H0 = A0["Hq"]; D = [info(np.eye(na)[j])["Hq"] - H0 for j in range(na)]
gi = np.flatnonzero(A0["isg"]); gi = gi[np.argsort(A0["e"][gi])]; T = np.zeros_like(H0)
for gk, tg in zip(gi, np.sort(targets)): T += (tg - A0["e"][gk]) * np.outer(A0["ph"][:, gk], A0["ph"][:, gk])
a_ls, *_ = np.linalg.lstsq(np.array([d.ravel() for d in D]).T, T.ravel(), rcond=None); lin_res = np.linalg.norm(sum(aj * d for aj, d in zip(a_ls, D)) - T)
def resid(a):
    r = info(a); return np.sort(r["e"][r["isg"]]) - targets if r["isg"].sum() == ng else np.full(ng, 1e3)
class _B: pass
best = _B(); best.x = a_ls
astar = best.x; A, B = info(np.zeros(na)), info(astar)
print(f"V/g={V:.3f} tc={tc} B_c={Bc} eps={EPS:g}: ED poles (w>1e-3) {np.round(sats,3)} weights {np.round(wsat,3)}; targets {np.round(targets,3)}; linear solve: |T - sum a_j D_j| = {lin_res:.1e}; found shift |a|={np.linalg.norm(astar):.2f} (residual of ghost levels {np.abs(resid(astar)).max():.1e})\n")
print("UNCHANGED: E =", f"{E:.10f}", "| nu_c =", f"{q[0]['nu']:.10f}", "| rho, R, Phi_I, Delta, hyb, lam_sum: from z (not touched)")
print(f"  QP pole {A['e'][np.argmax(A['wc'])]:.7f} -> {B['e'][np.argmax(B['wc'])]:.7f};  QP weight {A['wc'].max():.7f} -> {B['wc'].max():.7f};  total occ. core weight {A['wc'][A['occ']>0.5].sum():.8f} -> {B['wc'][B['occ']>0.5].sum():.8f}")
print(f"  occupations (dist. from 0/1) {np.minimum(abs(A['occ']),abs(1-A['occ'])).max():.1e} -> {np.minimum(abs(B['occ']),abs(1-B['occ'])).max():.1e}; |[H_qp,rho]| {np.linalg.norm(A['Hq']@rho-rho@A['Hq']):.1e} -> {np.linalg.norm(B['Hq']@rho-rho@B['Hq']):.1e}; aufbau {A['e'][A['occ']>0.5].max() < A['e'][A['occ']<=0.5].min()} -> {B['e'][B['occ']>0.5].max() < B['e'][B['occ']<=0.5].min()}")
print("\nCHANGED:")
print("  H_qp levels  a=0:", np.round(A['e'], 3), "occ", np.round(A['occ']).astype(int), "\n               a*:", np.round(B['e'], 3), "occ", np.round(B['occ']).astype(int))
print("  core weights a=0:", np.array2string(A['wc'], precision=1), "\n               a*:", np.array2string(B['wc'], precision=1))
Pg = lambda r: r['ph'][:, r['isg']] @ r['ph'][:, r['isg']].T
qa, qb = A['ph'][:, np.argmax(A['wc'])], B['ph'][:, np.argmax(B['wc'])]
print(f"  QP orbital overlap |<QP(a*)|QP(0)>| = {abs(qa @ qb):.6f};  ghost-subspace projector difference ||P_ghost(a*) - P_ghost(0)|| = {np.linalg.norm(Pg(A) - Pg(B)):.2e}  (0 = same ghost orbitals)")
print(f"  other (non-ghost) levels a=0 {np.round(np.sort(A['e'][~A['isg']]),3)}  a* {np.round(np.sort(B['e'][~B['isg']]),3)}")
print("  ghost levels: a=0", np.round(np.sort(A['e'][A['isg']]), 3), "core weights", np.array2string(A['wc'][A['isg']], precision=1), "-> a*", np.round(np.sort(B['e'][B['isg']]), 3), "core weights", np.array2string(B['wc'][B['isg']], precision=1))
dl = B['lam'][0] - A['lam'][0]; print(f"  lambda_c: max |change| of any entry {np.abs(dl).max():.2f} (a=0 diag {np.round(np.diag(A['lam'][0]),2)}, a* diag {np.round(np.diag(B['lam'][0]),2)}); lambda_o, lambda_v change: {abs(B['lam'][1]-A['lam'][1]).max():.1e}, {abs(B['lam'][2]-A['lam'][2]).max():.1e}")
print("  lambda^c_c eigenvalues (embedding bath levels) a=0:", np.round(np.linalg.eigvalsh(A['lamc'][0]), 3), " a*:", np.round(np.linalg.eigvalsh(B['lamc'][0]), 3))
print(f"  H_imp,c: Phi_c eigenstate residual {A['resid']:.1e} -> {B['resid']:.1e}; rank of Phi_c in H_imp,c spectrum {A['k']} -> {B['k']}; lowest eigenvalues a=0 {np.round(A['ev'][:3],2)}  a* {np.round(B['ev'][:3],2)}")
