"""Second-order test of a converged gGA solution: reduced Hessian of the Lagrangian.

At a KKT point (see check_stationarity.py) the solution is a strict local
minimum of the constrained energy if the Hessian of the Lagrangian, taken
w.r.t. the primal variables at fixed multipliers, is positive definite on the
tangent space of the constraints (second-order sufficient condition).

Primal variables x = (R^0, R^1 | Delta^0, Delta^1 | kappa | phi^0, phi^1):
  R^I     Neff numbers each
  Delta^I symmetric Neff x Neff (coordinates on the basis E_ij + E_ji, i<=j)
  kappa   Neff^2 rotations occupied -> virtual of the qp Slater determinant
          (rho = e^A C0 C0^T e^-A), i.e. the manifold of Psi
  phi^I   impurity state, all coefficients in the N_up = N_dn sector
Constraints c(x) = 0 (per fragment):
  c1 = D[phi] - S(Delta) R          (Neff)
  c2 = Delta + Dbb[phi] - |phi|^2 1  (Neff(Neff+1)/2)
  c3 = Delta - rho^II(kappa)         (Neff(Neff+1)/2)
  c4 = |phi|^2 - 1                   (1)
Lagrangian (multipliers V, lamc, lam, E fixed at the solution) as in
check_stationarity.py. All Hessian blocks are analytic, including the exact
second derivative of S(Delta) = sqrt(Delta(1-Delta)+shift) from the
Sylvester equation. Nonzero blocks: (R,R), (R,Delta), (R,kappa), (Delta,Delta),
(kappa,kappa), (phi,phi); the constraint Jacobian couples them.

Zero modes expected: the O(Neff) rotations of each fragment's ghost basis (a
gauge freedom) -> Neff(Neff-1)/2 per fragment.
"""

import sys

import numpy as np
from scipy.linalg import expm, null_space

import gga_dimer as g
import check_stationarity as cs


# ---------------- derivatives of S(Delta) = sqrt(Delta(1-Delta)+shift) ------------

class SFun:
    def __init__(self, Delta, shift):
        self.n, self.W = np.linalg.eigh(Delta)
        self.s = np.sqrt(self.n * (1.0 - self.n) + shift)
        self.w = (1.0 - self.n[:, None] - self.n[None, :]) / (self.s[:, None] + self.s[None, :])
        self.den = self.s[:, None] + self.s[None, :]

    def dS(self, B):
        return self.W @ ((self.W.T @ B @ self.W) * self.w) @ self.W.T

    def d2S(self, B1, B2):
        d1, d2 = self.dS(B1), self.dS(B2)
        rhs = -(B1 @ B2 + B2 @ B1) - (d1 @ d2 + d2 @ d1)
        return self.W @ ((self.W.T @ rhs @ self.W) / self.den) @ self.W.T


def basis_sym(N):
    out = []
    for i in range(N):
        for j in range(i, N):
            b = np.zeros((N, N))
            b[i, j] = 1.0
            b[j, i] = 1.0
            out.append(b)
    return out


def pair_index(N):
    return [(i, j) for i in range(N) for j in range(i, N)]


# ---------------- impurity-state derivatives -------------------------------------

def imp_grads(phi, ops):
    """Gradients wrt the sector coefficients of D_a, Dbb_ab and |phi|^2 (quadratic forms)."""
    Neff, sec = ops["Neff"], ops["sector"]
    x = cs.sector_vec(phi, ops)
    bu = ops["bath_up"]; bd = ops["bath_dn"]
    cu, cd = ops["c_pu"], ops["c_pd"]
    bux = [b @ x for b in bu]; bdx = [b @ x for b in bd]
    cux, cdx = cu @ x, cd @ x
    gD = []
    for a in range(Neff):
        gfull = 0.5 * (bu[a].T @ cux + cu.T @ bux[a] + bd[a].T @ cdx + cd.T @ bdx[a])
        gD.append(np.asarray(gfull)[sec])
    gDbb = {}
    for a in range(Neff):
        for b in range(Neff):
            gfull = 0.5 * (bu[a].T @ bux[b] + bu[b].T @ bux[a] + bd[a].T @ bdx[b] + bd[b].T @ bdx[a])
            gDbb[(a, b)] = np.asarray(gfull)[sec]
    return gD, gDbb


def build(p, ops, t, shift):
    """Assemble the Hessian of the Lagrangian H (nx x nx) and the constraint Jacobian J."""
    N = ops["Neff"]
    N2 = 2 * N
    d = len(ops["sector"])
    pairs = pair_index(N)
    npair = len(pairs)
    Bs = basis_sym(N)

    rho = p["C"] @ p["C"].T
    Hqp = np.zeros((N2, N2))
    for I in range(2):
        Hqp[I * N:(I + 1) * N, I * N:(I + 1) * N] = -p["lam"][I]
    Hqp[:N, N:] = -t * np.outer(p["R"][0], p["R"][1])
    Hqp[N:, :N] = -t * np.outer(p["R"][1], p["R"][0])
    e, W = np.linalg.eigh(Hqp)
    As = []
    for a in range(N):
        for i in range(N):
            As.append(np.outer(W[:, N + a], W[:, i]) - np.outer(W[:, i], W[:, N + a]))
    nk = len(As)

    # variable layout
    iR = lambda I: slice(I * N, (I + 1) * N)
    iD = lambda I: slice(2 * N + I * npair, 2 * N + (I + 1) * npair)
    ik = slice(2 * N + 2 * npair, 2 * N + 2 * npair + nk)
    iP = lambda I: slice(2 * N + 2 * npair + nk + I * d, 2 * N + 2 * npair + nk + (I + 1) * d)
    nx = 2 * N + 2 * npair + nk + 2 * d

    H = np.zeros((nx, nx))
    Sf = [SFun(p["Delta"][I], shift) for I in range(2)]
    V, R = p["V"], p["R"]

    # (R0,R1): -4 t rho01
    H[iR(0), iR(1)] = -4.0 * t * rho[:N, N:]
    H[iR(1), iR(0)] = H[iR(0), iR(1)].T
    dr = [A @ rho - rho @ A for A in As]  # first-order rho responses [A, rho]
    for m, drm in enumerate(dr):
        H[iR(0), 2 * N + 2 * npair + m] = (-4.0 * t * drm[:N, N:] @ R[1])
        H[iR(1), 2 * N + 2 * npair + m] = (-4.0 * t * drm[N:, :N] @ R[0])
    H[ik, iR(0)] = H[iR(0), ik].T
    H[ik, iR(1)] = H[iR(1), ik].T
    # (R_I, Delta_k): -4 dS[B_k] V
    for I in range(2):
        for k, B in enumerate(Bs):
            col = -4.0 * Sf[I].dS(B) @ V[I]
            H[iR(I), 2 * N + I * npair + k] = col
            H[2 * N + I * npair + k, iR(I)] = col
    # (Delta_k, Delta_l): -4 V^T d2S R
    for I in range(2):
        for k, Bk in enumerate(Bs):
            for l, Bl in enumerate(Bs):
                H[2 * N + I * npair + k, 2 * N + I * npair + l] = -4.0 * V[I] @ Sf[I].d2S(Bk, Bl) @ R[I]
    # (kappa, kappa): 2 Tr[Hqp d2rho],  d2rho = 1/2([A_m,[A_n,rho]] + [A_n,[A_m,rho]])
    for m in range(nk):
        for n in range(m, nk):
            Am, An = As[m], As[n]
            c1 = An @ dr[m] - dr[m] @ An
            c2 = Am @ dr[n] - dr[n] @ Am
            val = 2.0 * np.trace(Hqp @ (0.5 * (c1 + c2)))
            H[2 * N + 2 * npair + m, 2 * N + 2 * npair + n] = val
            H[2 * N + 2 * npair + n, 2 * N + 2 * npair + m] = val
    # (phi,phi): 2 (H_imp - E0) on the sector
    for I in range(2):
        Hs = cs.imp_sector_H(V[I], p["lamc"][I], ops)
        # multiplier of |phi|=1 is the eigenvalue of the given state (Rayleigh quotient);
        # equals the lowest eigenvalue when phi is the ground state
        E0 = p["phi"][I] @ Hs @ p["phi"][I]
        H[iP(I), iP(I)] = 2.0 * (Hs - E0 * np.eye(d))

    # constraint Jacobian
    nc = 2 * (N + 2 * npair + 1)
    J = np.zeros((nc, nx))
    row = 0
    for I in range(2):
        phi = p["phi"][I]
        gD, gDbb = imp_grads(phi, ops)
        # c1
        for a in range(N):
            J[row + a, iP(I)] = gD[a]
            J[row + a, iR(I)] = -((Sf[I].W * Sf[I].s) @ Sf[I].W.T)[a, :]
            for k, B in enumerate(Bs):
                J[row + a, 2 * N + I * npair + k] = -(Sf[I].dS(B) @ R[I])[a]
        row += N
        # c2
        for q, (a, b) in enumerate(pairs):
            J[row + q, 2 * N + I * npair + q] = 1.0
            gg = gDbb[(a, b)].copy()
            if a == b:
                gg = gg - 2.0 * phi
            J[row + q, iP(I)] = gg
        row += npair
        # c3
        for q, (a, b) in enumerate(pairs):
            J[row + q, 2 * N + I * npair + q] = 1.0
            for m in range(nk):
                J[row + q, 2 * N + 2 * npair + m] = -dr[m][I * N + a, I * N + b]
        row += npair
        # c4
        J[row, iP(I)] = 2.0 * phi
        row += 1
    return H, J, dict(nx=nx, N=N, d=d, npair=npair, nk=nk)


def constraint_values(p, ops, shift):
    N = ops["Neff"]
    rho = p["C"] @ p["C"].T
    out = []
    for I in range(2):
        Eloc, D, Dbb, nrm = cs.imp_terms(p["phi"][I], ops)
        S = cs.S_of(p["Delta"][I], shift)
        rhoII = rho[I * N:(I + 1) * N, I * N:(I + 1) * N]
        out.append(np.abs(D - S @ p["R"][I]).max())
        out.append(np.abs(p["Delta"][I] + Dbb - nrm * np.eye(N)).max())
        out.append(np.abs(p["Delta"][I] - rhoII).max())
        out.append(abs(nrm - 1.0))
    return max(out)


def unpack(p0, x, meta, ops):
    """Point p(x) from the coordinates x (for finite-difference validation)."""
    N, d, npair, nk = meta["N"], meta["d"], meta["npair"], meta["nk"]
    Bs = basis_sym(N)
    q = {k: ([a.copy() for a in v] if isinstance(v, list) else v.copy()) for k, v in p0.items()}
    for I in range(2):
        q["R"][I] = p0["R"][I] + x[I * N:(I + 1) * N]
        Dd = np.zeros((N, N))
        for k, B in enumerate(Bs):
            Dd += x[2 * N + I * npair + k] * B
        q["Delta"][I] = p0["Delta"][I] + Dd
        q["phi"][I] = p0["phi"][I] + x[2 * N + 2 * npair + nk + I * d: 2 * N + 2 * npair + nk + (I + 1) * d]
    kap = x[2 * N + 2 * npair: 2 * N + 2 * npair + nk]
    N2 = 2 * N
    Hqp = np.zeros((N2, N2))
    for I in range(2):
        Hqp[I * N:(I + 1) * N, I * N:(I + 1) * N] = -p0["lam"][I]
    Hqp[:N, N:] = -1.0 * np.outer(p0["R"][0], p0["R"][1])
    Hqp[N:, :N] = Hqp[:N, N:].T
    e, W = np.linalg.eigh(Hqp)
    A = np.zeros((N2, N2))
    m = 0
    for a in range(N):
        for i in range(N):
            A += kap[m] * (np.outer(W[:, N + a], W[:, i]) - np.outer(W[:, i], W[:, N + a]))
            m += 1
    q["C"] = expm(A) @ p0["C"]
    return q


def flip_inert_ghosts(p, ops, tol=1e-6):
    """Make the point feasible when some ghost orbitals are inert (n < tol or n > 1-tol):
    flip the impurity bath occupation of those Delta-eigen-directions (fill where n ~ 0,
    empty where n ~ 1, both spins), so that <d d^dag> = Delta holds there. The ghost is
    decoupled (R = V = 0 on it), so E_loc, D and all other constraints are unchanged; the
    new state is an EXCITED eigenstate of the same H_imp."""
    N = ops["Neff"]
    q = {k: ([a.copy() for a in v] if isinstance(v, list) else v.copy()) for k, v in p.items()}
    log = []
    for I in range(2):
        n, W = np.linalg.eigh(p["Delta"][I])
        x = cs.sector_vec(p["phi"][I], ops)
        for a in range(N):
            if n[a] < tol or n[a] > 1 - tol:
                u = W[:, a]
                for key in ("bath_up", "bath_dn"):
                    op = sum(u[b] * ops[key][b] for b in range(N))
                    x = (op.T if n[a] < tol else op) @ x   # create where n~0, annihilate where n~1
                log.append((I, a, "filled" if n[a] < tol else "emptied"))
        x = x / np.linalg.norm(x)
        q["phi"][I] = x[ops["sector"]]
        assert abs(np.linalg.norm(q["phi"][I]) - 1.0) < 1e-10, "flipped state left the N_up=N_dn sector"
    return q, log


def lagrangian_H(p, ops, t, shift, E_I):
    """L with fixed multipliers plus the normalization terms -E_I(|phi_I|^2-1)."""
    L = cs.Lfun(p, ops, t, shift)
    for I in range(2):
        L -= E_I[I] * (p["phi"][I] @ p["phi"][I] - 1.0)
    return L


def analyse(p, ops, t, shift, label):
    N = ops["Neff"]
    H, J, meta = build(p, ops, t, shift)
    print(f"  [{label}] variables = {meta['nx']}, constraints = {J.shape[0]}, "
          f"constraint violation at the point = {constraint_values(p, ops, shift):.2e}")
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-9 * sv[0]).sum())
    Z = null_space(J, rcond=1e-9)
    print(f"  Jacobian rank = {rank} (of {J.shape[0]}), tangent-space dimension = {Z.shape[1]}")
    Hs = Z.T @ H @ Z
    Hs = 0.5 * (Hs + Hs.T)
    ev = np.linalg.eigvalsh(Hs)
    scale = np.abs(ev).max()
    ngauge = N * (N - 1) // 2 * 2
    print(f"  reduced-Hessian eigenvalues: min = {ev[0]:.3e}, max = {ev[-1]:.3e}")
    zero = int((np.abs(ev) < 1e-7 * max(scale, 1.0)).sum())
    neg = int((ev < -1e-7 * max(scale, 1.0)).sum())
    pos = int((ev > 1e-7 * max(scale, 1.0)).sum())
    print(f"  negative: {neg},  ~zero: {zero} (gauge modes expected: {ngauge}),  positive: {pos}")
    print(f"  smallest 8 eigenvalues: {np.array2string(ev[:8], precision=3)}")
    return ev, (H, J, Z, meta)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: check_hessian.py <R.in> <L.in> <U>   (Ng = Neff-1 read from R.in)")
        sys.exit(1)
    R0, R1 = g.read_R_guess(sys.argv[1])
    L0, L1 = g.read_lambda_guess(sys.argv[2])
    U = float(sys.argv[3])
    t, shift = 1.0, 1e-9
    p, ops, res = cs.converged_point(R0, R1, L0, L1, U, t, shift, tol_mat=1e-8)
    print(f"converged: {res['converged']}  E_var = {res['E_var']:.10f}  Z = ({res['Z0']:.6f}, {res['Z1']:.6f})")
    analyse(p, ops, t, shift, "solution")
    if constraint_values(p, ops, shift) > 1e-6:
        print("\n  The point violates the constraints (inert ghost orbitals). Making it feasible by flipping the")
        print("  impurity occupation of the inert ghost bath orbitals (same E_loc, R, V; excited eigenstate of H_imp):")
        try:
            q, log = flip_inert_ghosts(p, ops)
        except AssertionError as err:
            print(f"   cannot be made feasible: {err}. (The inert ghosts would need net particle-number changes,")
            print("   i.e. the qp trace of Delta is incompatible with the impurity sector: no feasible point")
            print("   with this R exists in the N_up = N_dn sector, so no second-order test applies.)")
            sys.exit(0)
        print("   flipped:", log)
        Ea = [cs.imp_terms(q["phi"][I], ops)[0] for I in range(2)]
        print(f"   E_loc after flip: {sum(Ea):.10f}   (before: {sum(cs.imp_terms(p['phi'][I], ops)[0] for I in range(2)):.10f})")
        num, ana, agree = cs.analysis(q, ops, t, shift)
        print("   stationarity of L at the feasible point (max |dL/dx|, analytic):",
              {k: float(f"{v:.1e}") for k, v in ana.items()})
        analyse(q, ops, t, shift, "feasible forward point")


# ---------------- validation of the curvature along actual feasible curves ---------

def constraint_vector(q, ops, shift):
    N = ops["Neff"]
    pr = pair_index(N)
    out = []
    for I in range(2):
        Eloc, Dv, Dbb, nrm = cs.imp_terms(q["phi"][I], ops)
        S = cs.S_of(q["Delta"][I], shift)
        rho = q["C"] @ q["C"].T
        rII = rho[I * N:(I + 1) * N, I * N:(I + 1) * N]
        out += list(Dv - S @ q["R"][I])
        M2 = q["Delta"][I] + Dbb - nrm * np.eye(N)
        out += [M2[a, b] for a, b in pr]
        M3 = q["Delta"][I] - rII
        out += [M3[a, b] for a, b in pr]
        out.append(nrm - 1.0)
    return np.array(out)


def energy(q, ops, t):
    N = ops["Neff"]
    rho = q["C"] @ q["C"].T
    R0, R1 = q["R"]
    E = -2.0 * t * (R0 @ rho[:N, N:] @ R1 + R1 @ rho[N:, :N] @ R0)
    for I in range(2):
        E += cs.imp_terms(q["phi"][I], ops)[0]
    return E


def _jac_fd(p, ops, shift, meta, x):
    """Constraint Jacobian at x (fixed coordinates) by central differences."""
    nx = meta["nx"]
    N, npair = meta["N"], meta["npair"]
    Dlo, Dhi = 2 * N, 2 * N + 2 * npair
    cols = []
    for i in range(nx):
        # Delta step must stay below the shift (1e-9): at an eigenvalue n=0 (or 1) a larger step
        # makes n(1-n)+shift < 0 and sqrt() returns NaN
        h = 1e-10 if Dlo <= i < Dhi else 1e-6
        e = np.zeros(nx)
        e[i] = h
        cp_ = constraint_vector(unpack(p, x + e, meta, ops), ops, shift)
        cm_ = constraint_vector(unpack(p, x - e, meta, ops), ops, shift)
        cols.append((cp_ - cm_) / (2.0 * h))
    return np.array(cols).T


def feasible_curve_energy(p, ops, t, shift, meta, J0, v, s):
    """Energy at the feasible point nearest to s*v (Gauss-Newton, Jacobian by FD at each iterate)."""
    x = s * v
    for _ in range(40):
        q = unpack(p, x, meta, ops)
        c = constraint_vector(q, ops, shift)
        if not np.all(np.isfinite(c)):
            return np.nan, np.nan
        if np.abs(c).max() < 1e-11:
            break
        Jx = _jac_fd(p, ops, shift, meta, x)
        if not np.all(np.isfinite(Jx)):
            return np.nan, np.nan
        try:
            x = x - np.linalg.lstsq(Jx, c, rcond=1e-8)[0]
        except np.linalg.LinAlgError:
            return np.nan, np.nan
    else:
        return np.nan, np.abs(c).max()
    return energy(q, ops, t) - energy(p, ops, t), np.linalg.norm(x)


def curve_test(p, ops, t, shift, label, which):
    H, J, meta = build(p, ops, t, shift)
    Z = null_space(J, rcond=1e-9)
    Hs = Z.T @ H @ Z
    Hs = 0.5 * (Hs + Hs.T)
    ev, U = np.linalg.eigh(Hs)
    N, npair, nk, d = meta["N"], meta["npair"], meta["nk"], meta["d"]
    print(f"  [{label}] energy change along feasible curves x(s) = s*v + O(s^2), v = reduced-Hessian eigenvector")
    print(f"    {'eigenvalue':>11s} {'weights R/Delta/kappa/phi':>27s} {'s':>6s} {'dE actual':>12s} {'dE = lam s^2/2':>15s}")
    for k in which:
        v = Z @ U[:, k]
        v = v / np.linalg.norm(v)
        parts = [np.linalg.norm(v[:2 * N]) ** 2, np.linalg.norm(v[2 * N:2 * N + 2 * npair]) ** 2,
                 np.linalg.norm(v[2 * N + 2 * npair:2 * N + 2 * npair + nk]) ** 2,
                 np.linalg.norm(v[2 * N + 2 * npair + nk:]) ** 2]
        for s in (0.005, 0.01, 0.02, 0.05):
            dEp, _ = feasible_curve_energy(p, ops, t, shift, meta, J, v, s)
            dEm, _ = feasible_curve_energy(p, ops, t, shift, meta, J, -v, s)
            pred = 0.5 * ev[k] * s * s
            print(f"    {ev[k]:11.3f} {parts[0]:6.2f}/{parts[1]:4.2f}/{parts[2]:4.2f}/{parts[3]:4.2f} {s:6.2f} "
                  f"{dEp:12.3e} / {dEm:11.3e} {pred:12.3e}")
    return ev
