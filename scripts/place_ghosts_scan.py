"""Place the weightless ghost poles of G^h_cc on the exact (ED) satellites by the gauge shift, for every converged point of the eps=1e-7 scan
(V/g = 1/sqrt2, sqrt2; t_c/g = 0.2..5.0; B_c = 3, 5), and record everything that does / does not change.
The free empty ghost level (B_c=5) is placed on the second heaviest ED core particle-side pole. Other empty ghost levels below the highest occupied level (aufbau violation already present in the min-norm gauge at some B_c=5 points) are lifted to 0.3 above it.
Satellites: ED poles with weight > 1e-3 BELOW the main pole (largest weight), clustered when closer than 0.1 (weighted mean position, summed weight), sorted by weight;
B_c=3 (1 ghost) uses the heaviest cluster, B_c=5 (2 ghosts) the two heaviest (placed in energy order). If fewer clusters than ghosts exist, the remaining ghost keeps its min-norm position.
H_qp(a) = H_qp(0) + sum_j a_j D_j is exactly linear in the gauge parameters a (null directions of J_phi^T); the shift is the linear least-squares solution of
sum_j a_j D_j = sum_k (target_k - e_k) |g_k><g_k| over the ghost orbitals g_k at the min-norm gauge.
Output: cederbaum_data/ghost_placed/V{V}.npz (saved after every point), log printed. GHOST_EPS=1e-7 python3 scripts/place_ghosts_scan.py V"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g, exact_cederbaum as exm, not_imp_gga_cederbaum as N
from bc5_cederbaum import embed
EPS = float(os.environ.get("GHOST_EPS", "1e-7")); SUF = f"_eps{EPS:g}"; V = float(sys.argv[1])
OUT = "cederbaum_data/ghost_placed" + ("" if EPS == 1e-7 else f"_eps{EPS:g}"); os.makedirs(OUT, exist_ok=True); F = f"{OUT}/V{V}.npz"
store = dict(np.load(F)) if os.path.exists(F) else {}
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
def satellites(rr):
    p, w = rr["poles"], rr["weights"]; keep = w > 1e-3; p, w = p[keep], w[keep]; main = p[np.argmax(w)]
    idx = [i for i in np.argsort(p) if p[i] < main - 0.5]; cl = []
    for i in idx:
        if cl and abs(p[i] - cl[-1]["pos"][-1]) < 0.1: cl[-1]["pos"].append(p[i]); cl[-1]["w"].append(w[i])
        else: cl.append(dict(pos=[p[i]], w=[w[i]]))
    pos = np.array([np.average(c["pos"], weights=c["w"]) for c in cl]); ww = np.array([sum(c["w"]) for c in cl]); o = np.argsort(-ww)
    return main, pos[o], ww[o]
def particle_target(V, tc):
    """Second heaviest ED core particle-side pole (the heaviest is the v-like peak that gGut reproduces with its fixed v level); target of the free empty ghost level (B_c=5)."""
    H, c, n = exm.build_hamiltonian(1.0, 10.0, V, tc, ht="cv"); E, U = np.linalg.eigh(H); gs = U[:, 0]
    wp = (U.T @ (c[exm.C_UP].T @ gs)) ** 2; p = E - E[0]; i = np.argsort(-wp)[:2]
    return float(p[i[1]]), float(wp[i[1]])
data = np.load(f"cederbaum_data/scan_ghost_tc{SUF}/V{V}.npz")
for tc in ([float(x) for x in os.environ['GHOST_TCS'].split(',')] if os.environ.get('GHOST_TCS') else [round(0.2 * i, 2) for i in range(1, 26)]):
    rr = exm.solve(1.0, 10.0, V, tc, ht="cv"); main, spos, sw = satellites(rr)
    for Bc in (3, 5):
        key = f"tc{tc}_B{Bc}_"
        if key + "targets" in store or f"tc{tc}_B{Bc}_z" not in data: continue
        z = data[f"tc{tc}_B{Bc}_z"]
        if Bc == 3: m = g.Model(1.0, 10.0, V, tc, 3, EPS, "cv"); P = N.Problem(m)
        else: P, _ = embed(data[f"tc{tc}_B3_z"], V, tc, EPS); m = P.m
        E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z); J = P.cons_jac(z)[:, :P.nx]; o = N.recover(P, z)
        U_, s, _ = np.linalg.svd(J, full_matrices=True); rank = int((s > 1e-5 * s[0]).sum()); null = U_[:, rank:]; mu0, *_ = np.linalg.lstsq(J.T, -gz[:P.nx], rcond=None); na = null.shape[1]
        rho, R, Dl, q = st["rho"], st["R"], st["Dl"], st["q"]; alpha, Xi = m.core_extra(rho, R, Dl[0]); ng = (Bc - 1) // 2
        def info(a):
            mu = mu0 + null @ a; lam = [np.zeros((b, b)) for b in m.B]; row = 0
            for I in range(3):
                for i, j in zip(*P.iu[I]): lam[I][i, j] = lam[I][j, i] = (-mu[row] / 2.0 if i == j else -mu[row] / 4.0); row += 1
            Hq = m.build_Hqp(R, lam, Tb, u, w, o["nu"]); e, ph = np.linalg.eigh(Hq); occ = np.einsum("ik,ij,jk->k", ph, rho, ph); wc = (R[0] @ ph[m.sl[0], :]) ** 2
            blk0 = (ph[m.sl[0], :] ** 2).sum(0); qpi = int(np.argmax(wc)); isg = (occ > 0.5) & (blk0 > 0.9); isg[qpi] = False
            lamc = [ls[I] - lam[I] for I in range(3)]; Hi = himp(m, 0, hyb[0], lamc[0], alpha, Xi); ev, U = np.linalg.eigh(Hi); phi = st["phis"][0]
            return dict(lam=lam, e=e, ph=ph, occ=occ, wc=wc, qpi=qpi, isg=isg, blk0=blk0, Hq=Hq, rankPhi=int(np.argmax((U.T @ phi) ** 2)), residPhi=np.linalg.norm(Hi @ phi - (phi @ Hi @ phi) * phi), lamc_ev=np.linalg.eigvalsh(lamc[0]))
        A = info(np.zeros(na)); gi = np.flatnonzero(A["isg"]); gi = gi[np.argsort(A["e"][gi])]
        if len(gi) != ng or na == 0: print(f"V={V:.4f} tc={tc} B_c={Bc}: SKIP (ghosts found {len(gi)}, expected {ng}; free dirs {na})", flush=True); continue
        nt = min(ng, len(spos)); targets = np.sort(spos[:nt]); H0 = A["Hq"]; D = [info(np.eye(na)[j])["Hq"] - H0 for j in range(na)]; T = np.zeros_like(H0)
        for gk, tg in zip(gi[:nt], targets): T += (tg - A["e"][gk]) * np.outer(A["ph"][:, gk], A["ph"][:, gk])
        # free empty ghost level (B_c=5: the pure empty orbital): the level of the empty core block that responds to the gauge shifts -> placed on the ED particle-side pole;
        # other empty core-block ghost levels that lie below the highest occupied level would violate aufbau and are lifted to 0.3 above it.
        ge = np.flatnonzero((A["occ"] <= 0.5) & (A["blk0"] > 0.9)); eocc = A["e"][(A["occ"] > 0.5) & ~A["isg"]].max(); lifted = 0; free_empty = None; etarget = np.nan
        resp = [sum(abs(A["ph"][:, k] @ d @ A["ph"][:, k]) for d in D) for k in ge]
        for k, rk in zip(ge, resp):
            if rk > 1e-3 and Bc == 5:
                etarget, ew = particle_target(V, tc); free_empty = k; T += (etarget - A["e"][k]) * np.outer(A["ph"][:, k], A["ph"][:, k])
            elif A["e"][k] < eocc + 1e-6:
                T += (eocc + 0.3 - A["e"][k]) * np.outer(A["ph"][:, k], A["ph"][:, k]); lifted += 1
        a_ls, *_ = np.linalg.lstsq(np.array([d.ravel() for d in D]).T, T.ravel(), rcond=None); lin_res = np.linalg.norm(sum(aj * d for aj, d in zip(a_ls, D)) - T)
        B = info(a_ls); gB = np.flatnonzero(B["isg"]); gB = gB[np.argsort(B["e"][gB])]; Pg = lambda r, ix: r["ph"][:, ix] @ r["ph"][:, ix].T
        od = lambda r: np.minimum(abs(r["occ"]), abs(1 - r["occ"])).max(); cm = lambda r: np.linalg.norm(r["Hq"] @ rho - rho @ r["Hq"])
        empty = B["e"][B["occ"] <= 0.5]; occd = B["e"][B["occ"] > 0.5]
        store.update({key + "targets": targets, key + "ghost_before": np.sort(A["e"][gi]), key + "ghost_after": np.sort(B["e"][gB]) if len(gB) == ng else np.full(ng, np.nan), key + "lin_res": lin_res, key + "a": a_ls,
            key + "qp_before": A["e"][A["qpi"]], key + "qp_after": B["e"][B["qpi"]], key + "wq_before": A["wc"][A["qpi"]], key + "wq_after": B["wc"][B["qpi"]],
            key + "wocc_before": A["wc"][A["occ"] > 0.5].sum(), key + "wocc_after": B["wc"][B["occ"] > 0.5].sum(), key + "occdev_before": od(A), key + "occdev_after": od(B), key + "comm_before": cm(A), key + "comm_after": cm(B),
            key + "aufbau_after": bool(occd.max() < empty.min()), key + "ceiling": empty.min(), key + "ghostw_after": B["wc"][gB] if len(gB) == ng else np.full(ng, np.nan),
            key + "rankPhi_before": A["rankPhi"], key + "rankPhi_after": B["rankPhi"], key + "residPhi_before": A["residPhi"], key + "residPhi_after": B["residPhi"],
            key + "qp_overlap": abs(A["ph"][:, A["qpi"]] @ B["ph"][:, B["qpi"]]), key + "ghostproj_diff": np.linalg.norm(Pg(A, gi) - Pg(B, gB)) if len(gB) == ng else np.nan,
            key + "lamc_before": A["lamc_ev"], key + "lamc_after": B["lamc_ev"], key + "sats": spos, key + "satw": sw, key + "main": main, key + "E": E, key + "nu_c": q[0]["nu"], key + "nfree": na, key + "lifted": lifted,
            key + "levels_before": A["e"], key + "levels_after": B["e"], key + "occ": B["occ"], key + "ghost_rank_dim": A["Hq"].shape[0]})
        if free_empty is not None:
            kb = int(np.argmax(np.abs(B["ph"].T @ A["ph"][:, free_empty]))); store.update({key + "empty_before": A["e"][free_empty], key + "empty_target": etarget, key + "empty_after": B["e"][kb], key + "empty_target_w": ew})
        else: store.update({key + "empty_before": np.nan, key + "empty_target": np.nan, key + "empty_after": np.nan, key + "empty_target_w": np.nan})
        np.savez(F, **store)
        print(f"V={V:.4f} tc={tc} B_c={Bc}: targets {np.round(targets,3)} -> ghosts {np.round(store[key+'ghost_after'],3)} (lin.res {lin_res:.1e}); QP {A['e'][A['qpi']]:.6f}->{B['e'][B['qpi']]:.6f}; wQP {A['wc'][A['qpi']]:.6f}->{B['wc'][B['qpi']]:.6f}; "
              f"occdev {od(B):.1e}; comm {cm(A):.1e}->{cm(B):.1e}; aufbau {store[key+'aufbau_after']} (empty ghosts lifted: {lifted}; free empty ghost {store[key+'empty_before']:.3f} -> {store[key+'empty_after']:.3f}, target {store[key+'empty_target']:.3f}); rank(Phi_c) {A['rankPhi']}->{B['rankPhi']}", flush=True)
