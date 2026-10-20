"""Root-finder on the stationarity conditions of the Cederbaum gGut Lagrangian.

Unknowns x = (R_c,R_o,R_v, lam_c,lam_o,lam_v, nu, K, Delta_c, u, w); residuals are
one pass of steps 1-5 of the loop (Delta_I := rho^II as in gga_dimer) with the
fixed-point conditions made explicit:
  R' - R, (1-Dbb_I) - rho^II, nu' - nu, K' - K, Delta_c - rho^cc, u - rho^cv R_v, w - rho^co R_o.
Reuses Model/Fragment of gga_cederbaum.py. Random starts, scipy least_squares."""
import sys
import numpy as np
from scipy.optimize import least_squares
from gga_dimer import apply_S_inv
import gga_cederbaum as g


def tri(n):
    return np.triu_indices(n)


class Pack:
    def __init__(self, Bc):
        self.Bc = Bc
        self.sh = [("R", Bc), ("R", 1), ("R", 1), ("L", Bc), ("L", 1), ("L", 1), ("s", 1), ("L", Bc), ("L", Bc), ("v", Bc), ("v", Bc)]
        self.n = sum(k if t != "L" else k * (k + 1) // 2 for t, k in self.sh)

    def unpack(self, x):
        out, i = [], 0
        for t, k in self.sh:
            if t == "L":
                m = np.zeros((k, k)); iu = tri(k); c = k * (k + 1) // 2
                m[iu] = x[i:i + c]; m = m + m.T - np.diag(np.diag(m)); i += c
                out.append(m)
            else:
                out.append(x[i:i + k].copy() if t != "s" else x[i]); i += k
        return out

    def pack(self, parts):
        v = []
        for (t, k), p in zip(self.sh, parts):
            v.append(np.atleast_1d(p)[tri(k)] if t == "L" else np.atleast_1d(p))
        return np.concatenate(v)


def residual(x, m, pk):
    Rc, Ro, Rv, lc, lo, lv, nu, K, Dc, u, w = pk.unpack(x)
    R = [Rc, Ro, Rv]; lam = [lc, lo, lv]
    d, W = np.linalg.eigh(Dc)
    Dc_eff = (W * np.clip(d, 0.0, 1.0)) @ W.T      # keep S real; the residual Dc - rho^cc still uses Dc
    Tb = m.theta_bar(Dc_eff, nu, K)
    H = m.build_Hqp(R, lam, Tb, u, w, nu)
    _, _, rho = m.qp_state(H)
    D = [rho[s, s] for s in m.sl]
    u2, w2 = m.uw(rho, R)
    hyb = m.fit_hyb(rho, R, D, Tb, u2, w2, nu)
    ls = m.lam_sum(D, hyb, R, u2, w2, nu, K)
    lamc = [ls[i] - lam[i] for i in range(3)]
    alpha, Xi = m.core_extra(rho, R, D[0])
    imp = [m.frag[0].solve(hyb[0], lamc[0], alpha, Xi), m.frag[1].solve(hyb[1], lamc[1]), m.frag[2].solve(hyb[2], lamc[2])]
    res = []
    for I in range(3):
        Dn = np.eye(m.B[I]) - imp[I]["Dbb"]
        res += [apply_S_inv(Dn, imp[I]["D"], m.shift) - R[I], (Dn - D[I]).ravel()]
    res += [[imp[0]["nu"] - nu], (imp[0]["K"] - K).ravel(), (Dc - rho[m.sl[0], m.sl[0]]).ravel(), u - u2, w - w2]
    E = m.energy(rho, R, D[0], imp[0]["nu"], imp[0]["K"], sum(i["Eloc"] for i in imp))
    return np.concatenate([np.ravel(r) for r in res]), dict(E=E, nu=imp[0]["nu"], rho=rho, R=R, H=H, gap=[i["gap"] for i in imp])


def random_start(m, rng):
    Bc = m.Bc
    A = rng.standard_normal((Bc, Bc)); A = 0.5 * (A + A.T)
    eps = rng.uniform(-14, 12, Bc)
    if rng.random() < 0.6:      # informed start: filled core levels near the hole poles, empty ghost(s) between o and v
        nf = m.Nocc - 1
        eps = np.concatenate([np.sort(rng.uniform(-15, -7, nf)), rng.uniform(-0.8, 0.8, Bc - nf)])
    lc = -np.diag(eps) + rng.uniform(0, 2) * A
    q, _ = np.linalg.qr(rng.standard_normal((Bc, Bc)))
    Dc = (q * rng.uniform(0.02, 0.98, Bc)) @ q.T
    nu = rng.uniform(1.0, 2.0)
    K = 2 * nu * np.eye(Bc) / Bc + 0.1 * A
    parts = [rng.uniform(0.1, 1, Bc), rng.uniform(.3, 1, 1), rng.uniform(.3, 1, 1), lc,
             np.array([[m.Delta + 0.5 * rng.standard_normal()]]), np.array([[-m.Delta + 0.5 * rng.standard_normal()]]),
             nu, K, Dc, 0.05 * rng.standard_normal(Bc), 0.05 * rng.standard_normal(Bc)]
    return parts


if __name__ == "__main__":
    Delta, Ec, V, tc = 1.0, 10.0, float(sys.argv[1]), float(sys.argv[2])
    Bc, nst, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    m = g.Model(Delta, Ec, V, tc, Bc, shift=float(sys.argv[6]) if len(sys.argv) > 6 else 1e-9)
    pk = Pack(Bc); rng = np.random.default_rng(seed)
    print(f"unknowns {pk.n}; Bc={Bc} V={V} tc={tc}")
    sols = []
    for k in range(nst):
        x0 = pk.pack(random_start(m, rng))
        try:
            r = least_squares(lambda x: residual(x, m, pk)[0], x0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=3000)
            f, info = residual(r.x, m, pk)
            assert np.all(np.isfinite(f))
        except Exception as e:
            print(k, "fail", type(e).__name__); continue
        nr = np.abs(f).max()
        ev = np.linalg.eigvalsh(info["H"])[:m.Nocc]
        print(f"{k:3d} max|F|={nr:.1e} E={info['E']:.6f} nu={info['nu']:.6f} Rc2={info['R'][0]@info['R'][0]:.4f} eps={np.round(ev,3)}", flush=True)
        if nr < 1e-7: sols.append((info['E'], info['nu'], r.x))
    print("converged roots:", len(sols))
    np.save("roots_tmp.npy", np.array([s[2] for s in sols])) if sols else None
