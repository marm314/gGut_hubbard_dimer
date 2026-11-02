"""Intervals in which the weightless ghost poles can be moved (free multiplier directions = null space of the constraint-Jacobian transpose)
for every converged point of scan_ghost_tc. Valid shift: ghost count unchanged (occupied, core+ghost block >0.9, core weight <1e-2),
||[H_qp,rho]|| unchanged (<1e-6 above the min-norm value), QP pole unchanged (1e-4), QP weight and total occupied core weight unchanged (1e-5; the null directions are exact only to ~1e-7 in J^T d). Occupations n_k of H_qp levels must be as close to 0/1 as at the min-norm point (+1e-6). Rank cut: s > 1e-5 s0 (clean gap 1e-3 -> 1e-7). B_c=3: 1 free direction, scanned both ways until validity is lost.
B_c=5: 4 directions; axis scans + random-direction rays (sampled bounds, not exact). python3 scripts/ghost_range_tc.py V [tc]
Output cederbaum_data/scan_ghost_tc{SUF}/range_V{V}.npz"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed

EPS = float(os.environ.get("GHOST_EPS", "1e-9")); SUF = "" if EPS == 1e-9 else f"_eps{EPS:g}"; V = float(sys.argv[1]); only = [float(x) for x in sys.argv[2:]]
data = dict(np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")); out = {}
SMAX, NR = 400.0, 81


def setup(Bc, tc):
    z = data[f"tc{tc}_B{Bc}_z"]
    if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
    else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc, EPS); m = P.m
    o = N.recover(P, z); J = P.cons_jac(z)[:, :P.nx]; U, s, _ = np.linalg.svd(J, full_matrices=True)
    rank = int((s > 1e-5 * s[0]).sum()); null = U[:, rank:]
    E_, gz, st_, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None)
    def analyse(mu):
        lam = [np.zeros((bb, bb)) for bb in m.B]; row = 0
        for I in range(3):
            for a, bb in zip(*P.iu[I]):
                lam[I][a, bb] = lam[I][bb, a] = (-mu[row] / 2.0 if a == bb else -mu[row] / 4.0); row += 1
        H = m.build_Hqp(o["R"], lam, Tb, u, w, o["nu"]); e, phi = np.linalg.eigh(H)
        occ = np.einsum("ik,ij,jk->k", phi, o["rho"], phi); blk0 = (phi[m.sl[0], :] ** 2).sum(0); wc = (o["R"][0] @ phi[m.sl[0], :]) ** 2
        auf = e[occ > 0.5].max() < e[occ <= 0.5].min()          # aufbau: every rho-occupied level lies below every rho-empty level
        return e, (occ > 0.5) & (blk0 > 0.9) & (wc < 1e-2), np.linalg.norm(H @ o["rho"] - o["rho"] @ H), e[np.argmax(wc)], auf, wc.max(), wc[occ > 0.5].sum(), np.minimum(abs(occ), abs(1 - occ)).max(), np.sort(e[(occ > 0.5) & ~((blk0 > 0.9) & (wc < 1e-2))])
    return mu0, null, analyse, rank


for tc in [round(0.2 * i, 2) for i in range(1, 26)]:
    if only and tc not in only: continue
    for Bc in (3, 5):
        if f"tc{tc}_B{Bc}_z" not in data: continue
        mu0, null, analyse, rank = setup(Bc, tc); e0, isg0, c0, qp, _, wq0, wt0, od0, ng0e = analyse(mu0); dev = dict(occ=0.0, qp=0.0, wq=0.0, wt=0.0, comm=0.0, nong=0.0); ng = int(isg0.sum()); na = null.shape[1]; lev = []; laf = []
        def ray(d):          # contiguous valid interval around 0 along unit direction d (both signs); returns list of sorted ghost-level arrays
            res = []; ra = []
            for sign in (1, -1):
                for t in np.linspace(0, SMAX, NR)[1:]:
                    e, isg, c, qpk, au, wqk, wtk, odk, ngk = analyse(mu0 + sign * t * (null @ d))
                    if isg.sum() != ng or c > c0 + 1e-6 or abs(qpk - qp) > 1e-4 or abs(wqk - wq0) > 1e-5 or abs(wtk - wt0) > 1e-5 or odk > od0 + 1e-6: break
                    res.append(np.sort(e[isg]))
                    dev['occ'] = max(dev['occ'], odk); dev['qp'] = max(dev['qp'], abs(qpk - qp)); dev['wq'] = max(dev['wq'], abs(wqk - wq0)); dev['wt'] = max(dev['wt'], abs(wtk - wt0)); dev['comm'] = max(dev['comm'], abs(c - c0)); dev['nong'] = max(dev['nong'], np.abs(ngk - ng0e).max() if len(ngk) == len(ng0e) else 9e9)
                    if au: ra.append(np.sort(e[isg]))
            return res, ra
        lev.append(np.sort(e0[isg0])); laf.append(np.sort(e0[isg0]))
        dirs = list(np.eye(na)) + list(np.random.default_rng(1).standard_normal((60 if na > 1 else 0, na)))
        for d in dirs:
            r1, r2 = ray(d / np.linalg.norm(d)); lev += r1; laf += r2
        L = np.array(lev); LA = np.array(laf); k = f"tc{tc}_B{Bc}_"
        out[k + "lo"] = L.min(0); out[k + "hi"] = L.max(0); out[k + "lo_auf"] = LA.min(0); out[k + "hi_auf"] = LA.max(0); [out.__setitem__(k + "dev_" + q, v) for q, v in dev.items()]; out[k + "min_norm"] = lev[0]; out[k + "qp"] = qp; out[k + "ng"] = ng; out[k + "ndir"] = na
        print(f"V={V} tc={tc} B_c={Bc}: {ng} ghost(s), {na} free dir; min-norm {np.round(lev[0],2)}; worst |n_k-0/1| {dev["occ"]:.1e} (min-norm point {od0:.1e}); max dev from min-norm point (all accepted shifts): QP pole {dev["qp"]:.1e}, QP weight {dev["wq"]:.1e}, occ. core weight {dev["wt"]:.1e}, non-ghost occ. levels {dev["nong"]:.1e}, comm {dev["comm"]:.1e}; reachable lo {np.round(L.min(0),2)} hi {np.round(L.max(0),2)} | aufbau lo {np.round(LA.min(0),2)} hi {np.round(LA.max(0),2)}", flush=True)
        np.savez(f"cederbaum_data/scan_ghost_tc{SUF}/range_V{V}.npz", **out)
