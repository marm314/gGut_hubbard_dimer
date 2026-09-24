"""Ghost Gutzwiller for the half-filled Hubbard dimer WITHOUT the impurity problem.

Alternative to gga_dimer.py: instead of the iterative embedding cycle (impurity
Hamiltonian, V, lambda^c, fixed-point loop), the variational energy functional
of Tagliente, Pasqua, Fabrizio (NotImpurity.pdf, Eqs. 7, 9-12) is minimized
directly:

    E(psi*, phi) = sum_{i!=j} <psi*| f_i^dag R(phi_i)^dag t_ij R(phi_j) f_j |psi*>
                   + sum_i <phi_i| H_loc |phi_i>                       (Eq. 11)
    R(phi) = Q(phi) S(phi)^-1,   S^2 = Delta_phi (1 - Delta_phi),
    Q(phi) = <phi| c (x) f^dag |phi>                                    (Eqs. 9-10)

subject to the constraints (Eq. 7)

    <phi_i|phi_i> = 1,      Delta*(psi*)_ii = 1 - Delta_phi(phi_i),

the saddle point of F of Eq. 12 (the multipliers E*, E_phi, lambda are the
Lagrange multipliers of the normalizations and of the density constraint).

The problem is the same as in gga_dimer.py: half-filled dimer, paramagnetic
(spin-restricted: real phi_i in the N_up = N_dn sector, one density matrix for
both spins), the two fragments free to differ, H_loc = U n_up n_dn - (U/2) n,
Neff = 1 + Ng quasiparticle/bath orbitals per fragment and spin. In the
notation of gga_dimer.py:

    Delta_phi = Dbb = <phi| d^dag_a d_b |phi>  (spin average),
    Q_a       = D_a = <phi| d^dag_a c |phi>    (spin average),
    R_i       = S(1-Dbb)^-1 D_i,  S(Delta) = sqrt(Delta(1-Delta) + shift),

and E_kin = -4 t R_0 rho_01 R_1, with rho = X (X^T X)^-1 X^T the 1-RDM (per
spin) of the Slater determinant psi*: X is a (2 Neff x Neff) real matrix whose
column space is the occupied space (Neff of 2 Neff levels: half filling).

Variables: x_0, x_1 (unnormalized phi_i coefficients, phi_i = x_i/|x_i|) and X.
The constraints Delta*_ii = 1 - Dbb_i (Neff(Neff+1)/2 per fragment) are imposed
as equalities (SLSQP) with analytic gradients (checked against finite
differences by `--check-gradients`). After the minimization the multiplier
lambda is recovered from the stationarity of phi (linear least squares),
H_qp(R, lambda) is built, and the KKT conditions are verified through
[H_qp, rho] = 0 and through H_imp phi = E phi with V, lambda^c from Eqs. 7-8
of gga_dimer.py -- an independent check that the point is a solution of the
embedding equations too. Only the operator matrices of gga_dimer.py are reused;
gga_dimer.py itself is not modified and no impurity ground state is ever computed.
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize

import gga_dimer as g


def _F_inv_sqrt(m):
    """Divided difference (f(m_i)-f(m_j))/(m_i-m_j) of f(m) = m^-1/2, in the
    cancellation-free form -1/(sqrt(m_i) sqrt(m_j) (sqrt(m_i)+sqrt(m_j)))."""
    s = np.sqrt(m)
    return -1.0 / (s[:, None] * s[None, :] * (s[:, None] + s[None, :]))


class NotImpDimer:
    def __init__(self, Ng, U, t=1.0, shift=1e-9):
        self.Ng, self.U, self.t, self.shift = Ng, U, t, shift
        N = self.N = Ng + 1
        ops = g.build_impurity_ops(N, U, U / 2.0)
        sec = ops["sector"]
        self.m = len(sec)
        rs = lambda A: A.tocsr()[sec][:, sec].toarray()
        self.Hloc = rs(ops["H_loc"])
        self.Hyb = np.array([rs(ops["hyb_up"][a] + ops["hyb_dn"][a]) for a in range(N)])
        B = np.array([[0.5 * rs(ops["lam_up"][a][b] + ops["lam_dn"][a][b]) for b in range(N)]
                      for a in range(N)])
        self.B = 0.5 * (B + B.transpose(1, 0, 2, 3))        # symmetric in (a, b)
        self.Ndd = rs(ops["n_pu"] @ ops["n_pd"])
        self.Nph = rs(ops["n_pu"] + ops["n_pd"])
        self.iu = np.triu_indices(N)
        self.nsym = len(self.iu[0])
        self.ops = ops
        self.nz = 2 * self.m + 2 * N * N

    # ---- variables ------------------------------------------------------
    def unpack(self, z):
        m, N = self.m, self.N
        return z[:m], z[m:2 * m], z[2 * m:].reshape(2 * N, N)

    def pack(self, x0, x1, X):
        return np.concatenate([x0, x1, X.ravel()])

    def projector(self, X):
        Y = np.linalg.inv(X.T @ X)
        rho = X @ Y @ X.T
        return rho, Y

    # ---- one fragment -----------------------------------------------------
    def frag(self, x):
        nrm = np.linalg.norm(x)
        phi = x / nrm
        Bphi = np.einsum("abij,j->abi", self.B, phi)
        Dbb = np.einsum("i,abi->ab", phi, Bphi)
        Hphi = np.einsum("aij,j->ai", self.Hyb, phi)
        D = 0.25 * np.einsum("i,ai->a", phi, Hphi)
        Eloc = phi @ self.Hloc @ phi
        d, W = np.linalg.eigh(np.eye(self.N) - Dbb)
        mm = d * (1.0 - d) + self.shift
        f = mm ** -0.5
        R = W @ (f * (W.T @ D))
        return dict(nrm=nrm, phi=phi, Bphi=Bphi, Dbb=Dbb, D=D, Hphi=Hphi, Eloc=Eloc,
                    d=d, W=W, mm=mm, f=f, R=R)

    # ---- objective and gradient (Eq. 11) ----------------------------------
    def energy(self, z, grad=True):
        N, t = self.N, self.t
        x0, x1, X = self.unpack(z)
        rho, Y = self.projector(X)
        fr = [self.frag(x0), self.frag(x1)]
        R0, R1 = fr[0]["R"], fr[1]["R"]
        rho01 = rho[:N, N:]
        E = -4.0 * t * R0 @ rho01 @ R1 + fr[0]["Eloc"] + fr[1]["Eloc"]
        if not grad:
            return E
        G = np.zeros((2 * N, 2 * N))
        G[:N, N:] = -2.0 * t * np.outer(R0, R1)
        G[N:, :N] = G[:N, N:].T
        gX = 2.0 * (np.eye(2 * N) - rho) @ G @ X @ Y
        gR = [-4.0 * t * rho01 @ R1, -4.0 * t * rho01.T @ R0]
        gx = []
        for I in range(2):
            F = fr[I]
            u, v = F["W"].T @ gR[I], F["W"].T @ F["D"]
            gMt = np.outer(u, v) * _F_inv_sqrt(F["mm"])
            gMt = 0.5 * (gMt + gMt.T)
            dd = F["d"][:, None] + F["d"][None, :]
            GDelta = F["W"] @ (gMt * (1.0 - dd)) @ F["W"].T
            gDbb = -GDelta
            gD = F["W"] @ (F["f"] * u)
            gphi = (2.0 * np.einsum("ab,abi->i", gDbb, F["Bphi"])
                    + 0.5 * np.einsum("a,ai->i", gD, F["Hphi"])
                    + 2.0 * self.Hloc @ F["phi"])
            gx.append((gphi - (F["phi"] @ gphi) * F["phi"]) / F["nrm"])
        return E, self.pack(gx[0], gx[1], gX)

    # ---- constraints Delta*_ii + Dbb_i - 1 = 0 (Eq. 7) --------------------
    def constraints(self, z):
        N = self.N
        x0, x1, X = self.unpack(z)
        rho, _ = self.projector(X)
        out = []
        for I, x in enumerate((x0, x1)):
            Dbb = self.frag(x)["Dbb"]
            blk = rho[I * N:(I + 1) * N, I * N:(I + 1) * N]
            out.append((blk + Dbb - np.eye(N))[self.iu])
        return np.concatenate(out)

    def constraint_jac(self, z):
        N, m = self.N, self.m
        x0, x1, X = self.unpack(z)
        rho, Y = self.projector(X)
        P = np.eye(2 * N) - rho
        J = np.zeros((2 * self.nsym, self.nz))
        for I, x in enumerate((x0, x1)):
            F = self.frag(x)
            phi = F["phi"]
            for k, (a, b) in enumerate(zip(*self.iu)):
                row = I * self.nsym + k
                dphi = 2.0 * F["Bphi"][a, b]
                dphi = (dphi - (phi @ dphi) * phi) / F["nrm"]
                J[row, I * m:(I + 1) * m] = dphi
                G = np.zeros((2 * N, 2 * N))
                ia, ib = I * N + a, I * N + b
                if a == b:
                    G[ia, ia] = 1.0
                else:
                    G[ia, ib] = G[ib, ia] = 0.5
                J[row, 2 * m:] = (2.0 * P @ G @ X @ Y).ravel()
        return J

    # ---- starting point ----------------------------------------------------
    def random_start(self, rng, R_seed=None):
        N = self.N
        x0, x1 = rng.normal(size=self.m), rng.normal(size=self.m)
        R0 = np.zeros(N)
        R0[0] = 0.9
        if N > 1:
            R0[1:] = 0.05 * (1.0 + 0.3 * np.arange(N - 1))
        _, rho = g.qp_state([R0, R0], [np.zeros((N, N))] * 2, self.t)
        w, V = np.linalg.eigh(rho)
        X = V[:, -N:] + 0.05 * rng.normal(size=(2 * N, N))
        return self.pack(x0, x1, X)

    def start_from_R_lambda(self, R, lam):
        """Starting point from an (R, lambda) guess, as read by gga_dimer.py: X = occupied
        orbitals of H_qp(R, lambda); phi_i = lowest state of the sector Hamiltonian built from
        V_i, lambda^c_i of Eqs. 7-8 (one small diagonalization, used ONLY to initialize phi)."""
        N, t, shift = self.N, self.t, self.shift
        Hqp, rho = g.qp_state(R, lam, t)
        w, Vv = np.linalg.eigh(Hqp)
        X = Vv[:, :N]
        xs = []
        for I in range(2):
            Delta = rho[I * N:(I + 1) * N, I * N:(I + 1) * N]
            V = g.hybridization_V(rho, R, t, shift, I)
            lamc = g.embedding_lambda_c(Delta, lam[I], V, R[I], shift)
            H = self.Hloc.copy()
            for a in range(N):
                H = H + V[a] * self.Hyb[a]
                for b in range(N):
                    H = H + lamc[a, b] * (2.0 * self.B[a, b])
            # lowest eigenstates of H_imp: keep the one whose bath density best satisfies the
            # constraint Dbb = 1 - Delta_qp (the ground state itself does not when the seed has
            # inert ghosts with swapped occupations, e.g. the "forward" seed)
            ev, vec = np.linalg.eigh(H)
            best, best_err = None, np.inf
            for k in range(min(16, len(ev))):
                Dbb_k = np.einsum("i,abij,j->ab", vec[:, k], self.B, vec[:, k])
                err = np.abs(Dbb_k - (np.eye(N) - Delta)).max()
                if err < best_err - 1e-9:
                    best, best_err = k, err
            xs.append(vec[:, best])
        return self.pack(xs[0], xs[1], X)

    # ---- solve ---------------------------------------------------------------
    def minimize(self, z0, max_iter=1000, ftol=1e-13):
        return minimize(lambda z: self.energy(z), z0, jac=True, method="SLSQP",
                        constraints=[{"type": "eq", "fun": self.constraints, "jac": self.constraint_jac}],
                        options=dict(maxiter=max_iter, ftol=ftol))

    # ---- second-order test ------------------------------------------------------
    def hessian_analysis(self, z, h=1e-6):
        """Hessian of the Lagrangian  L = E + mu.c  (mu fixed by grad E + J^T mu = 0) with respect
        to all variables, by central differences of its analytic gradient, projected on the
        tangent space of the constraints (null space of the Jacobian J = dc/dz). Zero modes
        expected for Ng=2: 2 (norms of x_i) + N^2 = 9 (X -> X A) + 2 N(N-1)/2 = 6 (rotations of
        the ghost basis of each fragment) = 17; the rest must be > 0 for a strict local minimum."""
        E, g0 = self.energy(z)
        J = self.constraint_jac(z)
        mu = -np.linalg.lstsq(J.T, g0, rcond=None)[0]
        stat = np.abs(g0 + J.T @ mu).max()

        def gradL(zz):
            return self.energy(zz)[1] + self.constraint_jac(zz).T @ mu

        n = self.nz
        H = np.empty((n, n))
        for i in range(n):
            zp, zm = z.copy(), z.copy()
            zp[i] += h
            zm[i] -= h
            H[:, i] = (gradL(zp) - gradL(zm)) / (2.0 * h)
        asym = np.abs(H - H.T).max()
        H = 0.5 * (H + H.T)
        u, sv, vt = np.linalg.svd(J)
        rank = int((sv > 1e-9 * sv[0]).sum())
        Zt = vt[rank:].T
        ev = np.linalg.eigvalsh(Zt.T @ H @ Zt)
        return dict(E=E, mu=mu, stationarity=stat, rank=rank, ntangent=Zt.shape[1], eig=ev,
                    asym=asym, Hmax=np.abs(H).max())

    # ---- analysis of a solution ---------------------------------------------
    def analyse(self, z):
        N, t, shift = self.N, self.t, self.shift
        x0, x1, X = self.unpack(z)
        rho, Y = self.projector(X)
        E, grad = self.energy(z)
        fr = [self.frag(x0), self.frag(x1)]
        R = [fr[0]["R"], fr[1]["R"]]
        Delta = [np.eye(N) - fr[I]["Dbb"] for I in range(2)]

        # recover lambda_0, lambda_1 (12 numbers for Ng=2) from BOTH stationarity conditions at once:
        #   phi_I on the unit sphere:  grad_phi E - 4 sum_ab Lam_ab B_ab phi = mu phi
        #   psi*:                      [K - Lam, rho] = 0   (K = hopping part of H_qp)
        # (the phi condition alone is rank deficient when ghosts are inert, so a min-norm
        # solution of it alone need not make rho an eigenprojector of H_qp)
        ns = self.nsym
        _, gphi_full = self.energy(z)
        K = np.zeros((2 * N, 2 * N))
        K[:N, N:] = -t * np.outer(R[0], R[1])
        K[N:, :N] = K[:N, N:].T
        blocks, rhs = [], []
        for I in range(2):
            F = fr[I]
            phi = F["phi"]
            A = np.zeros((self.m, 2 * ns))
            for k, (a_, b_) in enumerate(zip(*self.iu)):
                v = 4.0 * (1.0 if a_ == b_ else 2.0) * F["Bphi"][a_, b_]
                A[:, I * ns + k] = v - (phi @ v) * phi
            blocks.append(A)
            rhs.append(gphi_full[I * self.m:(I + 1) * self.m] * F["nrm"])
        Ap = np.zeros(((2 * N) ** 2, 2 * ns))
        for I in range(2):
            for k, (a_, b_) in enumerate(zip(*self.iu)):
                Lam = np.zeros((2 * N, 2 * N))
                Lam[I * N + a_, I * N + b_] = 1.0
                Lam[I * N + b_, I * N + a_] = 1.0
                if a_ == b_:
                    Lam[I * N + a_, I * N + a_] = 1.0
                Ap[:, I * ns + k] = (Lam @ rho - rho @ Lam).ravel()
        bp = (K @ rho - rho @ K).ravel()
        A_all = np.vstack(blocks + [Ap])
        b_all = np.concatenate(rhs + [bp])
        sol = np.linalg.lstsq(A_all, b_all, rcond=1e-10)[0]
        kkt_phi = max(np.abs(blocks[I] @ sol - rhs[I]).max() for I in range(2))
        lam = []
        for I in range(2):
            L = np.zeros((N, N))
            L[self.iu] = sol[I * ns:(I + 1) * ns]
            lam.append(L + L.T - np.diag(np.diag(L)))

        Hqp, _ = g.qp_state(R, lam, t)
        kkt_psi = np.abs(Hqp @ rho - rho @ Hqp).max()
        eqp = np.linalg.eigvalsh(Hqp)

        # embedding quantities of gga_dimer.py (Eqs. 7, 8) and H_imp phi = E phi
        V = [g.hybridization_V(rho, R, t, shift, i) for i in range(2)]
        lamc = [g.embedding_lambda_c(Delta[i], lam[i], V[i], R[i], shift) for i in range(2)]
        kkt_imp = 0.0
        for I in range(2):
            H = self.Hloc.copy()
            for a in range(N):
                H = H + V[I][a] * self.Hyb[a]
                for b in range(N):
                    H = H + lamc[I][a, b] * (2.0 * self.B[a, b])
            phi = fr[I]["phi"]
            r = H @ phi - (phi @ H @ phi) * phi
            kkt_imp = max(kkt_imp, np.abs(r).max())

        docc = 0.5 * sum(phi_ @ self.Ndd @ phi_ for phi_ in (fr[0]["phi"], fr[1]["phi"]))
        nphys = 0.5 * sum(phi_ @ self.Nph @ phi_ for phi_ in (fr[0]["phi"], fr[1]["phi"]))
        return dict(Ng=self.Ng, U=self.U, E_var=E, R0=R[0], R1=R[1], Z0=float(R[0] @ R[0]),
                    Z1=float(R[1] @ R[1]), docc=docc, n_phys=nphys,
                    lam0=lam[0], lam1=lam[1], lamc0=lamc[0], lamc1=lamc[1], V0=V[0], V1=V[1],
                    Delta=Delta, rho=rho, eqp=eqp,
                    constraint=np.abs(self.constraints(z)).max(),
                    kkt_phi=kkt_phi, kkt_psi=kkt_psi, kkt_imp=kkt_imp,
                    phi=[fr[0]["phi"], fr[1]["phi"]], z=z)


def run_not_imp(Ng, U, t=1.0, nstarts=10, seed=0, shift=1e-9, max_iter=1000, z0=None,
                verbose=False, tol_constraint=1e-8, R=None, lam=None):
    """Direct minimization of the ghost-GA functional for the dimer. Returns the
    lowest feasible solution among `nstarts` random starts (or from z0)."""
    model = NotImpDimer(Ng, U, t, shift)
    rng = np.random.default_rng(seed)
    best = None
    if R is not None and lam is not None:
        z0 = model.start_from_R_lambda(R, lam)
    starts = [z0] if z0 is not None else [model.random_start(rng) for _ in range(nstarts)]
    for k, zs in enumerate(starts):
        res = model.minimize(zs, max_iter=max_iter)
        viol = np.abs(model.constraints(res.x)).max()
        if verbose:
            print(f"  start {k:3d}: E={res.fun: .8f}  |constraint|={viol:.2e}  iters={res.nit}  "
                  f"success={res.success}", flush=True)
        if viol < tol_constraint and (best is None or res.fun < best[0]):
            best = (res.fun, res.x)
    if best is None:
        raise RuntimeError("no start reached a feasible point (constraints not satisfied)")
    out = model.analyse(best[1])
    out["model"] = model
    return out


def check_gradients(Ng=2, U=2.0, seed=1, h=1e-6):
    model = NotImpDimer(Ng, U)
    rng = np.random.default_rng(seed)
    z = model.random_start(rng)
    E, gE = model.energy(z)
    idx = rng.choice(model.nz, size=25, replace=False)
    err = 0.0
    for i in idx:
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        err = max(err, abs((model.energy(zp, grad=False) - model.energy(zm, grad=False)) / (2 * h) - gE[i]))
    J = model.constraint_jac(z)
    errJ = 0.0
    for i in idx:
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        errJ = max(errJ, np.abs((model.constraints(zp) - model.constraints(zm)) / (2 * h) - J[:, i]).max())
    print(f"gradient check (Ng={Ng}, U={U}): max|dE/dz  analytic - FD| = {err:.2e},  "
          f"max|dc/dz analytic - FD| = {errJ:.2e}")


def main():
    p = argparse.ArgumentParser(description="ghost Gutzwiller for the Hubbard dimer by direct "
                                            "minimization of the variational functional (no impurity problem)")
    p.add_argument("--U", type=float)
    p.add_argument("--Ng", type=int)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--nstarts", type=int, default=10, help="random starts; the lowest feasible one is kept")
    p.add_argument("--R-file", default="R.in", help="R guess (gga_dimer.py format); if it and --lam-file exist "
                   "they are used as the (single) starting point instead of random starts")
    p.add_argument("--lam-file", default="L.in", help="lambda guess (gga_dimer.py format)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--shift", type=float, default=1e-9, help="shift in S = sqrt(D(1-D)+shift)")
    p.add_argument("--check-gradients", action="store_true", help="compare analytic and finite-difference gradients")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    if a.check_gradients:
        check_gradients(a.Ng if a.Ng is not None else 2, a.U if a.U is not None else 2.0)
        return
    if a.U is None or a.Ng is None:
        p.error("--U and --Ng are required")
    if a.Ng % 2 != 0:
        p.error("Ng must be even")

    R = lam = None
    if os.path.isfile(a.R_file) and os.path.isfile(a.lam_file):
        R, lam = list(g.read_R_guess(a.R_file)), list(g.read_lambda_guess(a.lam_file))
        print(f"[guess] starting from R = {a.R_file}, lambda = {a.lam_file}")
    r = run_not_imp(a.Ng, a.U, a.t, a.nstarts, a.seed, a.shift, a.max_iter, verbose=not a.quiet, R=R, lam=lam)
    print(f"Ng={r['Ng']}  U/t={r['U'] / a.t:.4f}  Z0={r['Z0']:.6f}  Z1={r['Z1']:.6f}  "
          f"E_var/t={r['E_var'] / a.t:.8f}  docc={r['docc']:.6f}")
    print(f"  constraints max|.|        = {r['constraint']:.2e}")
    print(f"  KKT: phi stationarity     = {r['kkt_phi']:.2e}   (lambda from phi and psi* conditions jointly)")
    print(f"  KKT: [H_qp, rho]          = {r['kkt_psi']:.2e}")
    print(f"  KKT: H_imp phi = E phi    = {r['kkt_imp']:.2e}   (V, lambda^c from Eqs. 7-8)")
    print(f"  eqp = {r['eqp']}")
    g.save_results(r, prefix=f"not_imp_Ng{r['Ng']}_U{r['U']:g}")


if __name__ == "__main__":
    main()
