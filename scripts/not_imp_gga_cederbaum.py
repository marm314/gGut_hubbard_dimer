"""Ghost Gutzwiller for the Cederbaum model by DIRECT minimization of the energy functional
(no embedding fixed-point loop), the analogue of not_imp_gga_dimer.py for three fragments c,o,v.

Minimize E of Eq. (Efunc) (doc/cederbaum.tex: c-v hopping by default) over
    phi_I   (real, normalized, in the half-filled sector of fragment I; x_I/|x_I| parametrization),
    X       (n x N_occ real matrix; rho = X (X^T X)^-1 X^T is the per-spin 1-RDM of psi_qp),
with  Delta_I = 1 - Dbb_I(phi_I),  R_I = S(Delta_I)^-1 D_I(phi_I),  nu_c, K from phi_c,
subject to  rho^{II} = Delta_I  (equality constraints, SLSQP, analytic gradients).

Gradients use the identities verified in check_cederbaum.py:
    dE = sum_I 4 hyb_I . dD_I + sum_I 2 Tr[lam_sum_I dDbb_I] + alpha dnu + Tr[Xi dK] + dEloc
    dE/drho = 2 H_qp(off-diagonal blocks),      (hyb = fit_hyb, lam_sum from Eq. Gdef/lamc3)
(`--check-gradients` compares them with central finite differences.)

After convergence the multipliers lambda_I are recovered from [H_off - Lambda, rho] = 0 (least
squares; its residual measures how well the point satisfies the KKT conditions), lambda^c = lam_sum - lambda,
and the hole poles/weights of G_cc, Eq. (GFgGut), are returned for comparison with ED.
"""
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

import gga_cederbaum as g
import exact_cederbaum as ed
from gga_dimer import apply_S_inv


class Problem:
    def __init__(self, m):
        self.m = m
        self.n, self.N = m.n, m.Nocc
        self.dim = [f.Hloc.shape[0] for f in m.frag]
        self.nx = sum(self.dim)
        self.iu = [np.triu_indices(b) for b in m.B]
        self.ncons = sum(len(i[0]) for i in self.iu)
        # The sector-projected operators are very sparse: store them as stacked CSR matrices so that all
        # B^2 bilinears <phi|d^dag_a d_b|phi> etc. come from a single sparse matrix-vector product.
        self.sp = []
        for I in range(3):
            f = m.frag[I]; B = f.B
            self.sp.append(dict(
                DD=sp.vstack([sp.csr_matrix(f.dd[a][b]) for a in range(B) for b in range(B)]).tocsr(),
                NCDD=sp.vstack([sp.csr_matrix(f.nc_dd[a][b]) for a in range(B) for b in range(B)]).tocsr(),
                DC=sp.vstack([sp.csr_matrix(f.dc[a]) for a in range(B)]).tocsr(),
                DCT=sp.vstack([sp.csr_matrix(f.dc[a].T) for a in range(B)]).tocsr(),
                nc=sp.csr_matrix(f.nc), Hloc=sp.csr_matrix(f.Hloc)))
        self._key = None

    # -- parameter handling
    def unpack(self, z):
        xs, p = [], 0
        for d in self.dim:
            xs.append(z[p:p + d]); p += d
        X = z[p:].reshape(self.n, self.N)
        return xs, X

    def pack(self, xs, X):
        return np.concatenate(list(xs) + [X.ravel()])

    def frag_quantities(self, I, phi):
        f = self.m.frag[I]; B = f.B; dim = phi.size; S = self.sp[I]
        T = (S["DD"] @ phi).reshape(B * B, dim)                 # dd_ab phi
        T2 = (S["NCDD"] @ phi).reshape(B * B, dim)              # n_c dd_ab phi
        nu = phi @ (S["nc"] @ phi)
        D = 0.5 * ((S["DC"] @ phi).reshape(B, dim) @ phi)
        Dbb = 0.5 * (T @ phi).reshape(B, B)
        K = 2.0 * nu * np.eye(B) - (T2 @ phi).reshape(B, B)
        return dict(nu=nu, D=D, Dbb=Dbb, K=K, Eloc=phi @ (S["Hloc"] @ phi), T=T, T2=T2)

    def state(self, z):
        key = z.tobytes()
        if self._key == key:
            return self._st
        st = self._state(z)
        self._key, self._st, self._eg, self._J = key, st, None, None
        return st

    def _state(self, z):
        m = self.m
        xs, X = self.unpack(z)
        phis = [x / np.linalg.norm(x) for x in xs]
        q = [self.frag_quantities(I, phis[I]) for I in range(3)]
        Dl = [np.eye(m.B[I]) - q[I]["Dbb"] for I in range(3)]
        R = [apply_S_inv(Dl[I], q[I]["D"], m.shift) for I in range(3)]
        Q, _ = np.linalg.qr(X)
        rho = Q @ Q.T
        return dict(xs=xs, X=X, phis=phis, q=q, Dl=Dl, R=R, rho=rho, Q=Q)

    def energy(self, st):
        m, q = self.m, st["q"]
        return m.energy(st["rho"], st["R"], st["Dl"][0], q[0]["nu"], q[0]["K"], sum(i["Eloc"] for i in q))

    def constraints(self, z):
        st = self.state(z)
        m = self.m
        c = []
        for I in range(3):
            r = st["rho"][m.sl[I], m.sl[I]] - st["Dl"][I]
            c.append(r[self.iu[I]])
        return np.concatenate(c)

    # -- gradients
    def energy_and_grad(self, z):
        st0 = self.state(z)
        if self._eg is not None:
            return self._eg
        m = self.m
        st = st0
        q, R, Dl, rho, phis = st["q"], st["R"], st["Dl"], st["rho"], st["phis"]
        nu, K = q[0]["nu"], q[0]["K"]
        u, w = m.uw(rho, R)
        Tb = m.theta_bar(Dl[0], nu, K)
        hyb = m.fit_hyb(rho, R, Dl, Tb, u, w, nu)
        ls = m.lam_sum(Dl, hyb, R, u, w, nu, K)
        alpha, Xi = m.core_extra(rho, R, Dl[0])
        E = self.energy(st)
        grads = []
        for I in range(3):
            f = m.frag[I]; B = f.B; phi = phis[I]; dim = phi.size; S = self.sp[I]; qI = q[I]
            gphi = 2.0 * (S["Hloc"] @ phi)
            Tdc = (S["DC"] @ phi).reshape(B, dim); Tdct = (S["DCT"] @ phi).reshape(B, dim)
            gphi = gphi + 2.0 * (hyb[I] @ (Tdc + Tdct))                 # 4 h_a 1/2 (dc_a + dc_a^T) phi
            gphi = gphi + ((ls[I] + ls[I].T).reshape(1, B * B) @ qI["T"]).ravel()   # sum_ab ls_ab (dd_ab + dd_ab^T) phi
            if I == 0:
                gphi = gphi + (alpha + 2.0 * np.trace(Xi)) * 2.0 * (S["nc"] @ phi)
                gphi = gphi - ((Xi + Xi.T).reshape(1, B * B) @ qI["T2"]).ravel()
            nrm = np.linalg.norm(st["xs"][I])
            grads.append((gphi - phi * (phi @ gphi)) / nrm)
        H_off = m.build_Hqp(R, [np.zeros((b, b)) for b in m.B], Tb, u, w, nu)
        G_rho = 2.0 * H_off
        X = st["X"]
        gX = 2.0 * (np.eye(self.n) - rho) @ G_rho @ X @ np.linalg.inv(X.T @ X)
        self._eg = (E, self.pack(grads, gX), st, (hyb, ls, Tb, u, w))
        return self._eg

    def cons_jac(self, z):
        st = self.state(z)
        if self._J is not None:
            return self._J
        m = self.m
        phis, X, rho = st["phis"], st["X"], st["rho"]
        J = np.zeros((self.ncons, z.size))
        XtXi = np.linalg.inv(X.T @ X)
        row = 0
        offs = np.cumsum([0] + self.dim)
        for I in range(3):
            B = m.frag[I].B
            nrm = np.linalg.norm(st["xs"][I]); phi = phis[I]; T = st["q"][I]["T"]
            for a, b in zip(*self.iu[I]):
                g_phi = 0.5 * (T[a * B + b] + T[b * B + a])             # 1/2 (dd_ab + dd_ab^T) phi
                J[row, offs[I]:offs[I + 1]] = (g_phi - phi * (phi @ g_phi)) / nrm
                W = np.zeros((self.n, self.n))
                i, j = m.sl[I].start + a, m.sl[I].start + b
                W[i, j] += 0.5; W[j, i] += 0.5
                gX = 2.0 * (np.eye(self.n) - rho) @ W @ X @ XtXi
                J[row, offs[3]:] = gX.ravel()
                row += 1
        self._J = J
        return J


def random_start(P, rng, scale_hyb=0.3):
    """phi_I from the ground states of random embedding Hamiltonians, X from a random H_qp-like matrix."""
    m = P.m
    xs = []
    for I in range(3):
        f = m.frag[I]; B = f.B
        hyb = rng.uniform(-scale_hyb, scale_hyb, B)
        A = rng.standard_normal((B, B)); A = 0.5 * (A + A.T)
        lc = A * rng.uniform(0.0, 6.0) + np.diag(rng.uniform(-8, 8, B))
        _, U = np.linalg.eigh(f.Hloc + sum(hyb[a] * f.hyb_op[a] for a in range(B)) +
                              sum(lc[a, b] * f.dd[a][b] for a in range(B) for b in range(B)))
        xs.append(U[:, 0] + 1e-3 * rng.standard_normal(f.Hloc.shape[0]))
    H = rng.standard_normal((P.n, P.n)); H = 0.5 * (H + H.T)
    _, V = np.linalg.eigh(H)
    return P.pack(xs, V[:, :P.N] + 1e-3 * rng.standard_normal((P.n, P.N)))


def recover(P, z):
    """Multipliers of the density constraints from the stationarity of E + sum_c mu_c c_c in phi
    (least squares; the residual `kkt_phi` measures how well the KKT conditions hold), with
    L = E + 2 Tr[lambda (Delta - rho^II)]  =>  lambda_aa = -mu_aa/2, lambda_ab = -mu_ab/4 (a<b).
    Then H_qp = H_off - Lambda; `comm` = ||[H_qp, rho]|| checks the stationarity in X."""
    m = P.m
    E, gz, st, (hyb, ls, Tb, u, w) = P.energy_and_grad(z)
    J = P.cons_jac(z)
    nx = P.nx
    mu, *_ = np.linalg.lstsq(J[:, :nx].T, -gz[:nx], rcond=None)
    kkt_phi = np.linalg.norm(J[:, :nx].T @ mu + gz[:nx])
    q, R, rho = st["q"], st["R"], st["rho"]
    nu = q[0]["nu"]
    lam = [np.zeros((b, b)) for b in m.B]
    row = 0
    for I in range(3):
        for a, b in zip(*P.iu[I]):
            lam[I][a, b] = lam[I][b, a] = (-mu[row] / 2.0 if a == b else -mu[row] / 4.0)
            row += 1
    H = m.build_Hqp(R, lam, Tb, u, w, nu)
    eps, phi = np.linalg.eigh(H)
    wts = np.array([(R[0] @ phi[m.sl[0], n]) ** 2 for n in range(m.Nocc)])
    comm = np.linalg.norm(H @ rho - rho @ H)
    return dict(E=E, nu=nu, R=R, Dl=st["Dl"], rho=rho, lam=lam, lamc=[ls[I] - lam[I] for I in range(3)],
                hyb=hyb, kkt=kkt_phi, comm=comm, eps=eps, poles=eps[:m.Nocc], weights=wts, st=st, Hqp=H)


def minimize_once(P, z0, maxiter=600):
    cons = {"type": "eq", "fun": P.constraints, "jac": P.cons_jac}
    res = minimize(lambda z: P.energy_and_grad(z)[0], z0, jac=lambda z: P.energy_and_grad(z)[1],
                   constraints=[cons], method="SLSQP", options=dict(maxiter=maxiter, ftol=1e-13))
    return res


def check_gradients(P, rng):
    z = random_start(P, rng)
    E, gz, *_ = P.energy_and_grad(z)
    Jz = P.cons_jac(z)
    worst_g = worst_j = 0.0
    for _ in range(12):
        v = rng.standard_normal(z.size)
        h = 1e-6
        fd = (P.energy_and_grad(z + h * v)[0] - P.energy_and_grad(z - h * v)[0]) / (2 * h)
        worst_g = max(worst_g, abs(fd - gz @ v))
        fdc = (P.constraints(z + h * v) - P.constraints(z - h * v)) / (2 * h)
        worst_j = max(worst_j, np.abs(fdc - Jz @ v).max())
    print(f"gradient check: max |dE_FD - g.v| = {worst_g:.2e} (|g| = {np.linalg.norm(gz):.2e});  "
          f"max |dC_FD - J.v| = {worst_j:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--Delta", type=float, default=1.0)
    p.add_argument("--Ec", type=float, default=10.0)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--tc", type=float, default=0.1)
    p.add_argument("--Bc", type=int, default=3, choices=[1, 3, 5])
    p.add_argument("--hop", default="cv", choices=["co", "cv"])
    p.add_argument("--starts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-gradients", action="store_true")
    a = p.parse_args()
    m = g.Model(a.Delta, a.Ec, a.V, a.tc, a.Bc, 1e-9, a.hop)
    P = Problem(m)
    rng = np.random.default_rng(a.seed)
    if a.check_gradients:
        check_gradients(P, rng)
    r = ed.solve(a.Delta, a.Ec, a.V, a.tc, ht=a.hop)
    print(f"ED: E0={r['E0']:.8f} nu_c={r['nu_c']:.6f}")
    for k in range(a.starts):
        res = minimize_once(P, random_start(P, rng))
        c = np.abs(P.constraints(res.x)).max()
        out = recover(P, res.x)
        print(f"start {k}: E={res.fun:.8f} (dE vs ED {res.fun-r['E0']:+.2e}) |cons|={c:.1e} nu={out['nu']:.6f} "
              f"KKT_phi={out['kkt']:.1e} |[Hqp,rho]|={out['comm']:.1e} Rc2={out['R'][0]@out['R'][0]:.4f} it={res.nit}\n"
              f"         poles {np.round(out['poles'],3)} w {np.round(out['weights'],4)}", flush=True)
