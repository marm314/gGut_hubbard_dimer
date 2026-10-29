"""Ghost Gutzwiller (gGut) for the Cederbaum model, three fragments c, o, v.

Implements Sec. "Self-consistent equations for the Cederbaum model" of main.tex
(and SI.tex) verbatim:

  fragments        c (B_c = 3 or 5 quasiparticle orbitals), o (B_o = 1), v (B_v = 1)
  H_loc            -E_c n_c,  -Delta n_o,  +Delta n_v
  energy           Eq. (Efunc),  with  Thetabar  of Eq. (ThetaBar)
  H_qp             Eq. (Hqp3)
  hybridization    Eq. (Vfit3)
  bath potential   G(Delta, M) of Eq. (Gdef) and, for the core, Eq. (lamc3)
  core embedding   Eq. (Himpc), built in normal order:
                   n_c [alpha + 2 Tr Xi - sum_{s,g,d} Xi_gd d^dag_gs d_ds]
  loop             Sec. "Self-consistency loop"

The shifted matrix square root S, the analytic derivative G (grad_S), the
Jordan-Wigner embedding operators (build_impurity_ops) and the linear mixer
are those of gga_dimer.py, generalized from two identical-size fragments to
three fragments of different size. The interaction is called Vint here, and
the hybridization Lagrange multiplier `hyb`, to avoid the clash of V with
the main text's V_I.

Conventions (as in the text): real, paramagnetic; rho is per spin; energies
are summed over spin. The quasiparticle orbitals are ordered [c_1..c_Bc, o, v].
"""

import numpy as np

from gga_dimer import _sqrt_eig, apply_S_inv, build_impurity_ops, grad_S


# ---------------------------------------------------------------------------
# Embedding fragments
# ---------------------------------------------------------------------------

class Fragment:
    """Embedding problem of one orbital: physical c + B bath orbitals per spin,
    searched in the half-filled sector N_up = N_dn = (1+B)/2 (B odd).

    All operators are projected on that sector and stored dense. `ops` and
    `sec` keep the full-Fock-space sparse operators for independent checks."""

    def __init__(self, B, mu):
        if B % 2 == 0:
            raise ValueError("B_I must be odd (half-filled embedding sector)")
        self.B = B
        self.mu = mu
        ops = build_impurity_ops(B, 0.0, mu)       # H_loc = -mu * n_c
        self.ops = ops
        self.sec = sec = ops["sector"]

        def proj(M):
            return M[sec][:, sec].toarray()

        c_pu, c_pd = ops["c_pu"], ops["c_pd"]
        self.Hloc = proj(ops["H_loc"])
        self.nc = proj(ops["n_pu"] + ops["n_pd"])
        # sum_s d^dag_{a s} c_s ; the hermitian conjugate is its transpose
        self.dc = [proj(ops["bath_up"][a].conj().T @ c_pu + ops["bath_dn"][a].conj().T @ c_pd)
                   for a in range(B)]
        self.hyb_op = [m + m.T for m in self.dc]
        # dd[a][b] = sum_s d^dag_{a s} d_{b s}
        self.dd = [[proj(ops["lam_up"][a][b] + ops["lam_dn"][a][b]) for b in range(B)]
                   for a in range(B)]
        self.nc_dd = [[self.nc @ self.dd[a][b] for b in range(B)] for a in range(B)]

    def solve(self, hyb, lamc, alpha=0.0, Xi=None, ref=None):
        """Ground state of H_imp (Eq. Himp0, or Eq. Himpc if Xi is given), in
        normal order. The normal-ordering constant -2 Tr(lamc) is dropped."""
        B = self.B
        H = self.Hloc.copy()
        for a in range(B):
            H += hyb[a] * self.hyb_op[a]
            for b in range(B):
                H += lamc[a, b] * self.dd[a][b]
        if Xi is not None:
            H += (alpha + 2.0 * np.trace(Xi)) * self.nc
            for g in range(B):
                for d in range(B):
                    H -= Xi[g, d] * self.nc_dd[g][d]
        E, U = np.linalg.eigh(H)
        k = 0
        if ref is not None:
            # maximum-overlap selection (excited-state SCF style): follow the eigenstate closest to `ref`
            # instead of the lowest one; degenerate subspaces are handled by projecting `ref` on them
            ov = (U.T @ ref) ** 2
            k = int(np.argmax(ov))
            deg = np.flatnonzero(np.abs(E - E[k]) < 1e-9)
            if len(deg) > 1:
                v = U[:, deg] @ (U[:, deg].T @ ref)
                U = U.copy(); U[:, k] = v / np.linalg.norm(v)
        gs = U[:, k]
        D = np.array([0.5 * gs @ self.dc[a] @ gs for a in range(B)])
        Dbb = np.array([[0.5 * gs @ self.dd[a][b] @ gs for b in range(B)] for a in range(B)])
        nu = gs @ self.nc @ gs
        # K_gd = <n_c sum_s d_ds d^dag_gs> = 2 delta_gd nu - <n_c sum_s d^dag_gs d_ds>
        K = 2.0 * nu * np.eye(B) - np.array([[gs @ self.nc_dd[g][d] @ gs for d in range(B)]
                                             for g in range(B)])
        return {"gs": gs, "E": E[k], "index": k, "gap": E[1] - E[0], "D": D, "Dbb": Dbb, "nu": nu, "K": K,
                "Eloc": gs @ self.Hloc @ gs}


# ---------------------------------------------------------------------------
# Quasiparticle side: Thetabar, E, H_qp, V fit, bath potentials
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, Delta, Ec, Vint, tc, Bc, shift=1e-9, hop="cv", Bo=1, Bv=1):
        # hop: which weak hopping H_t = t_c (c^dag_c c_x + h.c.) is added: x = o ('co', main.tex)
        # or x = v ('cv', doc/cederbaum.tex: the c-o hopping is Pauli-blocked in |220>)
        if hop not in ("co", "cv"):
            raise ValueError("hop must be 'co' or 'cv'")
        self.hop = hop
        self.Delta, self.Ec, self.Vint, self.tc, self.Bc, self.shift = Delta, Ec, Vint, tc, Bc, shift
        self.Bo, self.Bv = Bo, Bv                    # ghosts on o and v (default 1 = none); used by the direct-minimization solver only
        self.B = [Bc, Bo, Bv]
        self.n = Bc + Bo + Bv
        self.sl = [slice(0, Bc), slice(Bc, Bc + Bo), slice(Bc + Bo, Bc + Bo + Bv)]
        self.Nocc = 2 + (Bc + Bo + Bv - 3) // 2
        self.frag = [Fragment(Bc, Ec), Fragment(Bo, Delta), Fragment(Bv, -Delta)]

    # -- Thetabar, Eq. (ThetaBar):  1/2 S^-1 K S^-1 - nu (1 - Delta + eps)^-1
    def Cmat(self, Dc):
        d, W, _ = _sqrt_eig(Dc, self.shift)
        return (W / (1.0 - d + self.shift)) @ W.T

    def Sinv(self, Dc):
        d, W, s = _sqrt_eig(Dc, self.shift)
        return (W / s) @ W.T

    def theta_bar(self, Dc, nu, K):
        Si = self.Sinv(Dc)
        return 0.5 * Si @ K @ Si - nu * self.Cmat(Dc)

    def uw(self, rho, R):
        c, o, v = self.sl
        return rho[c, v] @ R[2], rho[c, o] @ R[1]

    # -- Eq. (Efunc)
    def energy(self, rho, R, Dc, nu, K, Eloc):
        c, o, v = self.sl
        u, w = self.uw(rho, R)
        Tb = self.theta_bar(Dc, nu, K)
        V = self.Vint
        rx, sx = (o, 1) if self.hop == "co" else (v, 2)
        return (Eloc + 4.0 * self.tc * R[0] @ rho[c, rx] @ R[sx]
                - 4.0 * V * (2.0 - nu) * R[2] @ rho[v, o] @ R[1]
                - 4.0 * V * w @ Tb @ u)

    # -- Eq. (Hqp3)
    def build_Hqp(self, R, lam, Tb, u, w, nu):
        c, o, v = self.sl
        V, tc = self.Vint, self.tc
        H = np.zeros((self.n, self.n))
        for I in range(3):
            H[self.sl[I], self.sl[I]] = -lam[I]
        H[c, o] = -V * np.outer(Tb @ u, R[1])
        H[c, v] = -V * np.outer(Tb @ w, R[2])
        if self.hop == "co":
            H[c, o] += tc * np.outer(R[0], R[1])
        else:
            H[c, v] += tc * np.outer(R[0], R[2])
        H[o, v] = -V * (2.0 - nu) * np.outer(R[1], R[2])
        H[o, c], H[v, c], H[v, o] = H[c, o].T.copy(), H[c, v].T.copy(), H[o, v].T.copy()
        return H

    def qp_state(self, H):
        e, phi = np.linalg.eigh(H)
        occ = np.zeros(self.n)
        occ[:self.Nocc] = 1.0
        return e, phi, (phi * occ) @ phi.T

    # -- Eq. (Vfit3)
    def fit_hyb(self, rho, R, D, Tb, u, w, nu):
        """Returns [V_c, V_o, V_v] (S_I V_I = ...)."""
        c, o, v = self.sl
        V, tc = self.Vint, self.tc
        X = [np.zeros(len(R[0])),
             -V * ((2.0 - nu) * rho[o, v] @ R[2] + rho[o, c] @ Tb @ u),
             -V * ((2.0 - nu) * rho[v, o] @ R[1] + rho[v, c] @ Tb @ w)]
        if self.hop == "co":
            X[0] = X[0] + tc * rho[c, o] @ R[1]
            X[1] = X[1] + tc * rho[o, c] @ R[0]
        else:
            X[0] = X[0] + tc * rho[c, v] @ R[2]
            X[2] = X[2] + tc * rho[v, c] @ R[0]
        return [apply_S_inv(D[I], X[I], self.shift) for I in range(3)]

    # -- bath potentials: returns lambda_I + lambda^c_I, Eq. (Gdef) / Eq. (lamc3)
    def lam_sum(self, D, hyb, R, u, w, nu, K):
        out = []
        for I in (1, 2):
            out.append(grad_S(D[I], np.outer(hyb[I], R[I]), self.shift))
        Dc = D[0]
        Si, C, V = self.Sinv(Dc), self.Cmat(Dc), self.Vint
        x, y = Si @ w, Si @ u
        Pi = np.outer(x, y) + np.outer(y, x)
        Y = V * (Si @ K @ Pi + Pi @ K @ Si)
        lc = (grad_S(Dc, np.outer(hyb[0], R[0]) - 0.25 * Y, self.shift)
              - V * nu * C @ (np.outer(u, w) + np.outer(w, u)) @ C)
        return [lc, out[0], out[1]]

    # -- Eq. (Himpc) parameters
    def core_extra(self, rho, R, Dc):
        c, o, v = self.sl
        u, w = self.uw(rho, R)
        Si, C, V = self.Sinv(Dc), self.Cmat(Dc), self.Vint
        x, y = Si @ w, Si @ u
        Pi = np.outer(x, y) + np.outer(y, x)
        alpha = 4.0 * V * (R[2] @ rho[v, o] @ R[1] + w @ C @ u)
        return alpha, -V * Pi


# ---------------------------------------------------------------------------
# Self-consistency loop
# ---------------------------------------------------------------------------

def default_guess(m):
    """Start consistent with the sign structure of the stationarity conditions (bath level
    = +lambda^c, lambda^c = -lambda + G): an isolated filled orbital needs its qp level ABOVE the
    bare one (eps > bare, else the embedding puts the bath electron in the wrong state), an empty
    one below it. So: eps_o = -Delta + 0.5, eps_v = Delta - 0.5, the main core level just above
    -E_c, further filled core ghosts below it, empty core ghosts between eps_o and eps_v."""
    Bc = m.Bc
    nfill = m.Nocc - 1                              # filled core levels
    eps = np.concatenate([-m.Ec + 0.8 - 2.0 * np.arange(nfill),
                          np.linspace(-0.3, 0.3, Bc - nfill) if Bc > nfill else []])
    Rc = np.concatenate([0.9 / np.sqrt(1 + np.arange(nfill)), 0.1 * np.ones(Bc - nfill)])
    R = [Rc, np.ones(1), np.ones(1)]
    lam = [-np.diag(eps), np.array([[m.Delta - 0.5]]), np.array([[-m.Delta + 0.5]])]
    return R, lam


def run(Delta, Ec, Vint, tc, Bc, R0=None, lam0=None, max_iter=500, tol=1e-8, mix=0.5, eq_time=0,
        shift=1e-9, delta_from_qp=True, verbose=False, hop="cv", follow=False, phi0=None, state0=None):
    """Self-consistency loop, steps 1-5 of Sec. "Self-consistency loop".

    State variables are R_I and ^I Delta (mixed linearly, eigenvalues of Delta
    clipped to [0,1] as in gga_dimer), plus lambda_I at iteration 0.
    delta_from_qp=True (default) replaces ^I Delta by rho^{II} after step 2, as gga_dimer
    does; False is the literal loop (^I Delta is the mixed state variable), which was unstable.
    follow=True selects, in each embedding solve, the eigenstate of maximum overlap with the previous one
    (phi0: starting reference states) instead of the lowest eigenstate.
    state0 (dict R, lam, nu, K, rho, hyb, lamc, Dl) starts the loop from a full previous state
    (all of iteration >= 1 is then used from the first iteration on), e.g. a direct-minimization solution.
    Converged when the changes of E_var, ^I Delta, R_I and the constraint
    residual ||rho^{II} - (1 - Delta^bb_I)|| are all below `tol`."""
    m = Model(Delta, Ec, Vint, tc, Bc, shift, hop)
    Rg, lg = default_guess(m)
    R = [np.array(r, float) for r in (R0 if R0 is not None else Rg)]
    lam = [np.array(l, float) for l in (lam0 if lam0 is not None else lg)]

    nu, K = 2.0, np.zeros((Bc, Bc))
    hyb = [np.zeros(b) for b in m.B]
    lamc = [np.zeros((b, b)) for b in m.B]
    rho = np.zeros((m.n, m.n))
    Dl = None
    E_prev = 100.0
    converged, hist = False, []
    ref = list(phi0) if (follow and phi0 is not None) else [None, None, None]
    first = 0
    if state0 is not None:
        R = [np.array(r, float) for r in state0["R"]]; lam = [np.array(l, float) for l in state0["lam"]]
        nu, K = float(state0["nu"]), np.array(state0["K"], float)
        rho = np.array(state0["rho"], float); hyb = [np.array(h, float) for h in state0["hyb"]]
        lamc = [np.array(l, float) for l in state0["lamc"]]; Dl = [np.array(d, float) for d in state0["Dl"]]
        first = 1

    for it in range(max_iter):
        # u, w from the previous quasiparticle density matrix, current R
        u, w = m.uw(rho, R)

        # (1) lambda_I
        if it == 0 and first == 0:
            # iteration 0: initial guess for lambda, nu = 2, Thetabar = 0
            Dc_for_theta = np.eye(Bc)           # irrelevant, K = 0
            Tb = np.zeros((Bc, Bc))
            nu, K = 2.0, np.zeros((Bc, Bc))
        else:
            ls = m.lam_sum(Dl, hyb, R, u, w, nu, K)
            lam = [ls[I] - lamc[I] for I in range(3)]
            Tb = m.theta_bar(Dl[0], nu, K)

        # (2) Thetabar -> H_qp -> rho
        Hqp = m.build_Hqp(R, lam, Tb, u, w, nu)
        eps_qp, phi, rho = m.qp_state(Hqp)
        if (it == 0 and first == 0) or delta_from_qp:
            Dl = [rho[m.sl[I], m.sl[I]].copy() for I in range(3)]
        u, w = m.uw(rho, R)

        # (3) hybridization and lambda^c
        hyb = m.fit_hyb(rho, R, Dl, Tb, u, w, nu)
        ls = m.lam_sum(Dl, hyb, R, u, w, nu, K)
        lamc = [ls[I] - lam[I] for I in range(3)]

        # (4) embedding problems
        alpha, Xi = m.core_extra(rho, R, Dl[0])
        imp = [m.frag[0].solve(hyb[0], lamc[0], alpha, Xi, ref=ref[0]),
               m.frag[1].solve(hyb[1], lamc[1], ref=ref[1]),
               m.frag[2].solve(hyb[2], lamc[2], ref=ref[2])]
        if follow:
            ref = [i["gs"] for i in imp]
        nu, K = imp[0]["nu"], imp[0]["K"]

        # (5) update, energy, mixing
        Dnew = [np.eye(m.B[I]) - imp[I]["Dbb"] for I in range(3)]
        Rnew = [apply_S_inv(Dnew[I], imp[I]["D"], shift) for I in range(3)]
        Eloc = sum(imp[I]["Eloc"] for I in range(3))
        E_var = m.energy(rho, R, Dl[0], nu, K, Eloc)
        resid = sum(np.linalg.norm(rho[m.sl[I], m.sl[I]] - Dnew[I]) for I in range(3))

        a = 1.0 if it < eq_time else mix
        nextD = [Dl[I] + a * (Dnew[I] - Dl[I]) for I in range(3)]
        nextR = [R[I] + a * (Rnew[I] - R[I]) for I in range(3)]
        for I in range(3):
            d, W = np.linalg.eigh(nextD[I])
            nextD[I] = (W * np.clip(d, 0.0, 1.0)) @ W.T
        dD = sum(np.linalg.norm(nextD[I] - Dl[I]) for I in range(3))
        dR = sum(np.linalg.norm(nextR[I] - R[I]) for I in range(3))
        dE = E_var - E_prev
        hist.append((it, E_var, dE, dD, dR, resid))
        if verbose:
            print(f"  it={it:4d} E={E_var:.10f} dE={dE:.2e} dDelta={dD:.2e} dR={dR:.2e} "
                  f"resid={resid:.2e} nu_c={nu:.6f}", flush=True)

        # everything stored belongs to this iteration (R = the one used in H_qp)
        sol = dict(hop=hop, Delta=Delta, Ec=Ec, Vint=Vint, tc=tc, Bc=Bc, shift=shift,
                   R=[r.copy() for r in R], Dl=[d.copy() for d in Dl], lam=[l.copy() for l in lam],
                   lamc=[l.copy() for l in lamc], hyb=[h.copy() for h in hyb], rho=rho.copy(),
                   Hqp=Hqp.copy(), eps=eps_qp, phi=phi, Tb=Tb.copy(), u=u, w=w, nu=nu, K=K.copy(),
                   imp=imp, Dnew=Dnew, Rnew=Rnew, E_var=E_var, resid=resid, alpha=alpha, Xi=Xi)

        R, Dl, E_prev = nextR, nextD, E_var
        if it > 0 and abs(dE) < tol and dD < tol and dR < tol and resid < tol:
            converged = True
            break

    sol.update(converged=converged, iters=it + 1, hist=hist)
    sol["poles"], sol["weights"] = hole_poles(m, sol)
    return sol


def hole_poles(m, sol, tol_w=1e-10):
    """Occupied qp levels and their core weights |R_c^T phi_n^(c)|^2, Eq. (GFgGut)."""
    eps, phi, Rc = sol["eps"], sol["phi"], sol["R"][0]
    w = np.array([(Rc @ phi[m.sl[0], n]) ** 2 for n in range(m.Nocc)])
    return eps[:m.Nocc], w


def noisy(lam, scale, rng):
    """lambda + small symmetric random noise."""
    out = []
    for l in lam:
        e = scale * rng.standard_normal(l.shape)
        out.append(l + 0.5 * (e + e.T))
    return out


if __name__ == "__main__":
    import argparse
    import exact_cederbaum as ed

    p = argparse.ArgumentParser(description="gGut for the Cederbaum model")
    p.add_argument("--Delta", type=float, default=1.0)
    p.add_argument("--Ec", type=float, default=10.0)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--tc", type=float, default=0.1)
    p.add_argument("--Bc", type=int, default=3, choices=[3, 5])
    p.add_argument("--mix", type=float, default=0.5)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--hop", default="cv", choices=["co", "cv"])
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    s = run(a.Delta, a.Ec, a.V, a.tc, a.Bc, mix=a.mix, max_iter=a.max_iter, verbose=not a.quiet, hop=a.hop)
    r = ed.solve(a.Delta, a.Ec, a.V, a.tc, ht=a.hop)
    print(f"converged={s['converged']} iters={s['iters']}  E_var={s['E_var']:.8f}  "
          f"E_ED={r['E0']:.8f}  nu_c={s['nu']:.6f} (ED {r['nu_c']:.6f})  resid={s['resid']:.2e}")
    print("gGut hole poles / weights:")
    for e, w in zip(*s["poles":] if False else (s["poles"], s["weights"])):
        print(f"  {e:12.6f} {w:12.6e}")
    print("ED hole poles / weights:")
    for e, w in zip(r["poles"], r["weights"]):
        if w > 1e-6:
            print(f"  {e:12.6f} {w:12.6e}")
