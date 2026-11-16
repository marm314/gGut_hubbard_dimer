"""gGut by DIRECT minimization (no impurity/embedding loop) for the FULL 3-site c-o-v model of hubbard_sat_data:
  H = sum_i eps_i n_i + sum_{i<j,s} t_ij (c+_is c_js + h.c.) + sum_i U_i n_iu n_id - V sum_{s,s'} (1-n_cs)(v+_s' o_s' + h.c.)
(same convention as fci_trimer.ham, Vc=V). Built on gga_cederbaum.Model / not_imp_gga_cederbaum.Problem without modifying them:
  - eps_i enter the fragment H_loc (mu_i = -eps_i), U_i are added to each fragment H_loc (as ucore_cederbaum.model_U does for the core),
  - the hoppings t_cv, t_co, t_ov enter as 4 t R_I rho_IJ R_J in E, t outer(R_I,R_J) in H_qp and t rho_IJ R_J in the hybridization fit
    (exactly the structure of the existing c-v hopping, generalized to every pair).
Ng=0 <-> B_c=1, Ng=2 <-> B_c=3 (B_o=B_v=1). Usage: python3 trimer_gut.py check | run <x.npy|literal> V Bc nstarts"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
from gga_dimer import apply_S_inv
from fci_trimer import spectrum

PAIRS = {"co": (0, 1), "ov": (1, 2), "cv": (0, 2)}


class TrimerModel(g.Model):
    """Model with arbitrary eps_i, U_i, t_co, t_ov, t_cv and the Cederbaum coupling V (= Vint)."""
    def __init__(self, eps, t, U, V, Bc, Bo=1, Bv=1, shift=1e-9):
        super().__init__(0.0, 0.0, V, 0.0, Bc, shift, "cv", Bo, Bv)
        self.eps, self.t, self.U = list(eps), dict(t), list(U)         # t = {"co":..,"ov":..,"cv":..}
        self.frag = [g.Fragment(b, -e) for b, e in zip(self.B, eps)]    # H_loc = eps n
        for f, u in zip(self.frag, U):
            ops = f.ops
            f.Hloc = f.Hloc + u * (ops["n_pu"] @ ops["n_pd"])[f.sec][:, f.sec].toarray()
        self.hops = [(PAIRS[k][0], PAIRS[k][1], v) for k, v in t.items() if v != 0.0]

    def energy(self, rho, R, Dc, nu, K, Eloc):
        E = super().energy(rho, R, Dc, nu, K, Eloc)                    # base: Eloc, V terms (its own tc = 0)
        for I, J, t in self.hops:
            E += 4.0 * t * R[I] @ rho[self.sl[I], self.sl[J]] @ R[J]
        return E

    def build_Hqp(self, R, lam, Tb, u, w, nu):
        H = super().build_Hqp(R, lam, Tb, u, w, nu)
        for I, J, t in self.hops:
            B = t * np.outer(R[I], R[J])
            H[self.sl[I], self.sl[J]] += B
            H[self.sl[J], self.sl[I]] += B.T
        return H

    def fit_hyb(self, rho, R, D, Tb, u, w, nu):
        hyb = super().fit_hyb(rho, R, D, Tb, u, w, nu)
        X = [np.zeros(b) for b in self.B]
        for I, J, t in self.hops:
            X[I] = X[I] + t * rho[self.sl[I], self.sl[J]] @ R[J]
            X[J] = X[J] + t * rho[self.sl[J], self.sl[I]] @ R[I]
        return [hyb[I] + apply_S_inv(D[I], X[I], self.shift) for I in range(3)]


def fci(eps, t, U, V):
    T = [[0, t["co"], t["cv"]], [t["co"], 0, t["ov"]], [t["cv"], t["ov"], 0]]
    E0, dg, n, om, wt = spectrum(list(eps), T, list(U), Vc=V)
    ip = -om; wc = wt[:, 0]; tot = wc.sum(); k = wc > 1e-3
    o = np.argsort(ip[k])
    return dict(E0=E0, n=n, ip=ip[k][o], w=wc[k][o] / tot, wabs=wc[k][o], ndeg=dg)


def solve(eps, t, U, V, Bc, nstarts=20, seed=0, tol=1e-3, ref=None, log=print, ctol=1e-9):
    m = TrimerModel(eps, t, U, V, Bc); P = N.Problem(m); rng = np.random.default_rng(seed); best = None
    for i in range(nstarts):
        res = N.minimize_once(P, N.random_start(P, rng), maxiter=600); cons = np.abs(P.constraints(res.x)).max(); ok = False
        if cons < ctol:
            o = N.recover(P, res.x); ok = o["comm"] < 1e-4
            log(f"      comm={o['comm']:.1e} kkt={o['kkt']:.1e} IP {np.round(-o['poles'],3)} w {np.array2string(o['weights'],precision=4)}")
            if ok and (best is None or res.fun < best[0].fun): best = (res, o)
        log(f"  Bc={Bc} start {i}: E={res.fun:.6f} cons={cons:.1e} accepted={ok}")
        if best is not None and ref is not None and best[0].fun - ref < tol: break
    return m, P, best


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":     # gradients of E and constraints (FD) for the full model; U=0,V=0 energy vs FCI (exact there)
        eps = [-7.941, -4.541, 0.013]; t = dict(co=0.05, ov=0.143, cv=0.163); U = [3.689, 4.104, 3.531]
        for Bc in (1, 3):
            P = N.Problem(TrimerModel(eps, t, U, 0.5, Bc)); N.check_gradients(P, np.random.default_rng(1))
        t0 = dict(co=0.5, ov=0.7, cv=0.3); e0 = [-3.0, -1.0, 0.5]
        r = fci(e0, t0, [0, 0, 0], 0.0); m, P, best = solve(e0, t0, [0, 0, 0], 0.0, 1, 10, ref=r["E0"], tol=1e-8)
        print(f"U=0,V=0,Bc=1: FCI E0={r['E0']:.8f}  gGut E={best[0].fun:.8f}  diff={best[0].fun-r['E0']:.2e}")
    elif mode == "run":
        x = np.array([float(a) for a in sys.argv[2].split(",")]); V, Bc, ns = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        eps, t, U = x[:3], dict(co=x[3], ov=x[4], cv=x[5]), x[6:9]
        r = fci(eps, t, U, V); sel = r["wabs"] > 1e-3
        print(f"FCI: E0={r['E0']:.6f} n={np.round(r['n'],4)} c-lines IP {np.round(r['ip'],3)} w/tot {np.round(r['w'],4)}", flush=True)
        t0 = time.time(); m, P, best = solve(eps, t, U, V, Bc, ns, ref=r["E0"], ctol=float(sys.argv[6]) if len(sys.argv) > 6 else 1e-9)
        if best is None: print("no accepted solution"); sys.exit()
        res, o = best; wts = o["weights"]; ipg = -o["poles"]; od = np.argsort(ipg)
        print(f"gGut B_c={Bc}: E-E_FCI={res.fun-r['E0']:.3e} nu_c={o['nu']:.5f} kkt={o['kkt']:.1e} comm={o['comm']:.1e} "
              f"IP {np.round(ipg[od],3)} w {np.round(wts[od],4)} (w/sum {np.round(wts[od]/wts.sum(),4)}) ({time.time()-t0:.0f}s)")
        np.savez("../hubbard_sat_data/trimer_gut_Bc%d.npz" % Bc, z=res.x, E=res.fun, E_FCI=r["E0"], ip=ipg, w=wts, x=x, V=V)
