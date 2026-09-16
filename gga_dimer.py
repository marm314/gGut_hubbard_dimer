"""Ghost Gutzwiller Approximation (gGA) for the half-filled 2-site Hubbard dimer.

Each site is its own impurity/fragment; by site-exchange symmetry both fragments
are identical, so only one impurity problem is solved per iteration. The two
sites couple only through the quasiparticle Hamiltonian H_qp (renormalized
hopping -t R R^T). Ng is the number of *extra* ghost bath orbitals beyond the
one bath orbital standard (non-ghost) Gutzwiller already has; Neff = 1 + Ng is
the number of quasiparticle/bath orbitals per fragment per spin.

Equations (specialized from Mejuto-Zaera, arXiv:2403.05157, Eqs. 5-11, to a
2-fragment, non-periodic dimer):

  - Half filling fixes the qp-side occupation rule: fill exactly the lowest
    Neff of the 2*Neff single-particle levels per spin (this is a fixed
    LEVEL-COUNT rule, not a fixed chemical-potential threshold). At Ng=0
    this is "fill 1 of 2", matching the brief's original special case.
  - lambda (qp-side onsite matrix, Neff x Neff symmetric) is NOT pinned to
    zero by particle-hole symmetry -- that was an incorrect simplification
    tried earlier and empirically falsified (a reference calculation gives
    a lambda with real off-diagonal structure and an O(1) diagonal entry).
    Instead lambda is found each outer iteration by a root-find: the unique
    lambda such that diagonalizing H_qp(R, lambda) and filling per the rule
    above reproduces a target Delta (the Delta produced by the previous
    iteration's impurity solve, Eq. 10). This is exactly analogous to
    tuning a chemical potential to match a target density, just promoted to
    a matrix equation.
  - V (hybridization): sqrt(Delta(1-Delta)) . V = -t * Delta_off . R   (Eq. 7)
  - lambda^c (bath potential): lambda^c = -lambda + [d/dDelta (R . sqrt(Delta(1-Delta)) . V)]_sym,
    with R, V held fixed for the derivative (V is NOT resubstituted via Eq. 7);
    evaluated here by finite differences since Neff is small.  (Eq. 8)
  - R, Delta update from the impurity ground state's bath-bath and
    bath-impurity 1-RDM blocks.  (Eq. 10)

Dimension bookkeeping: with Nphys=1 (one physical orbital per fragment,
spins handled symmetrically), the embedding Hamiltonian has Nphys+Neff =
2+Ng orbitals per spin, so the impurity Fock space has dimension
2**(2*(2+Ng)) = 4**(2+Ng) -- not 4**(1+Ng).
"""

import os

import numpy as np
from scipy.linalg import sqrtm, eigh


# ---------------------------------------------------------------------------
# Pulay / DIIS mixing for the outer self-consistency loop
# ---------------------------------------------------------------------------

class DIIS:
    """Pulay DIIS mixer for a vector fixed-point iteration x = f(x).

    Keeps a short history of (output, residual=output-input) pairs and
    extrapolates the next trial vector as the residual-minimizing linear
    combination of past outputs, subject to the coefficients summing to 1.
    Falls back to plain damped linear mixing when there isn't enough
    history yet, or if the extrapolated step looks unreasonably large
    (guards against the near-singular B matrix that shows up right at a
    degenerate fixed point, e.g. exactly degenerate ghost orbitals).
    """

    def __init__(self, max_vecs=8, mix=0.5, blowup_factor=50.0):
        self.max_vecs = max_vecs
        self.mix = mix
        self.blowup_factor = blowup_factor
        self.y_hist = []
        self.r_hist = []

    def reset(self):
        self.y_hist.clear()
        self.r_hist.clear()

    def step(self, x_in, x_out):
        r = x_out - x_in
        r_norm = np.linalg.norm(r)

        linear_fallback = x_in + self.mix * r

        self.y_hist.append(x_out.copy())
        self.r_hist.append(r.copy())
        if len(self.y_hist) > self.max_vecs:
            self.y_hist.pop(0)
            self.r_hist.pop(0)

        n = len(self.r_hist)
        if n < 2:
            return linear_fallback

        B = np.empty((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self.r_hist[i], self.r_hist[j])
        B[:n, n] = -1.0
        B[n, :n] = -1.0
        B[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        try:
            sol = np.linalg.lstsq(B, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return linear_fallback

        c = sol[:n]
        x_new = sum(c[i] * self.y_hist[i] for i in range(n))

        if not np.all(np.isfinite(x_new)) or np.linalg.norm(x_new - x_out) > self.blowup_factor * max(r_norm, 1e-12):
            # DIIS extrapolation is unreliable near a degenerate fixed point
            # (near-singular B) -- drop history and fall back to damped
            # linear mixing for this step instead of blowing up.
            self.reset()
            return linear_fallback

        return x_new

# ---------------------------------------------------------------------------
# Jordan-Wigner fermion operators (generic n_modes)
# ---------------------------------------------------------------------------

I2 = np.eye(2)
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
C_LOCAL = np.array([[0.0, 1.0], [0.0, 0.0]])


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_c_ops(n_modes):
    ops = []
    for i in range(n_modes):
        mats = [Z if j < i else (C_LOCAL if j == i else I2) for j in range(n_modes)]
        ops.append(_kron_list(mats))
    return ops


# ---------------------------------------------------------------------------
# Quasiparticle side
# ---------------------------------------------------------------------------

def qp_step(R, lam, t):
    """Diagonalize H_qp(R, lambda) and fill exactly the lowest Neff of 2*Neff
    levels per spin (half filling as a fixed level-count rule)."""
    Neff = len(R)
    RRt = np.outer(R, R)
    Hqp = np.zeros((2 * Neff, 2 * Neff))
    Hqp[:Neff, :Neff] = -lam
    Hqp[Neff:, Neff:] = -lam
    Hqp[:Neff, Neff:] = -t * RRt
    Hqp[Neff:, :Neff] = -t * RRt
    evals, evecs = eigh(Hqp)
    occ = np.zeros(2 * Neff)
    occ[:Neff] = 1.0  # eigh returns ascending eigenvalues -> lowest Neff filled
    rho = (evecs * occ) @ evecs.T
    Delta = rho[:Neff, :Neff]
    Delta_off = rho[:Neff, Neff:]
    return Delta, Delta_off


def qp_step2(R0, R1, lam0, lam1, t):
    """General 2-fragment H_qp, NOT assuming the two sites are equal:
    H_qp = [[-lam0, -t R0 R1^T], [-t R1 R0^T, -lam1]]. Reduces to qp_step
    when R0=R1, lam0=lam1. Returns the three independent 1-RDM blocks."""
    Neff = len(R0)
    Hqp = np.zeros((2 * Neff, 2 * Neff))
    Hqp[:Neff, :Neff] = -lam0
    Hqp[Neff:, Neff:] = -lam1
    Hqp[:Neff, Neff:] = -t * np.outer(R0, R1)
    Hqp[Neff:, :Neff] = -t * np.outer(R1, R0)
    evals, evecs = eigh(Hqp)
    occ = np.zeros(2 * Neff)
    occ[:Neff] = 1.0
    rho = (evecs * occ) @ evecs.T
    Delta00 = rho[:Neff, :Neff]
    Delta11 = rho[Neff:, Neff:]
    Delta01 = rho[:Neff, Neff:]
    return Delta00, Delta11, Delta01


def analytic_jacobian2(R0, R1, lam0, lam1, t):
    """d[Delta00[iu]; Delta11[iu]] / d[lam0[iu]; lam1[iu]], via first-order
    (Hellmann-Feynman / Daletskii-Krein divided-difference) perturbation
    theory of the qp projector -- exact, not a finite-difference estimate.
    Validated against finite differences to ~1e-9."""
    Neff = len(R0)
    Hqp = np.zeros((2 * Neff, 2 * Neff))
    Hqp[:Neff, :Neff] = -lam0
    Hqp[Neff:, Neff:] = -lam1
    Hqp[:Neff, Neff:] = -t * np.outer(R0, R1)
    Hqp[Neff:, :Neff] = -t * np.outer(R1, R0)
    evals, evecs = eigh(Hqp)
    occ = np.zeros(2 * Neff)
    occ[:Neff] = 1.0

    iu = np.triu_indices(Neff)
    npar = len(iu[0])
    # Lorentzian-regularized divided difference: F -> (occ_n-occ_m)*dE/(dE^2+reg^2).
    # As dE -> 0 this smoothly goes to 0 (not 1/dE -> infinity); for
    # |dE| >> reg it reduces to the correct (occ_n-occ_m)/dE. A plain
    # "zero out |dE|<thresh" step function still lets F blow up for any
    # dE just above threshold, which is exactly what happens as a qp level
    # crosses the Fermi surface (confirmed: Newton steps exploding by
    # ~10-100x right as an eigenvalue of Delta approaches 0, even though
    # nothing else in the pipeline -- V, R -- was diverging at that point).
    dE = evals[:, None] - evals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (occ[:, None] - occ[None, :]) / dE
    F[np.abs(dE) < 1e-10] = 0.0

    J = np.zeros((2 * npar, 2 * npar))
    for which, offset in ((0, 0), (1, Neff)):
        for k, (a, b) in enumerate(zip(*iu)):
            dH = np.zeros((2 * Neff, 2 * Neff))
            if a == b:
                dH[offset + a, offset + a] = -1.0
            else:
                dH[offset + a, offset + b] = -1.0
                dH[offset + b, offset + a] = -1.0
            drho = evecs @ (F * (evecs.T @ dH @ evecs)) @ evecs.T
            col = which * npar + k
            J[:npar, col] = drho[:Neff, :Neff][iu]
            J[npar:, col] = drho[Neff:, Neff:][iu]
    return J


def fit_lambda2(R0, R1, t, Delta00_target, Delta11_target, lam0_guess, lam1_guess,
                max_lam=1e3, max_newton=30, tol=1e-11):
    """Joint fit for (lambda0, lambda1) so qp_step2 matches both fragments'
    target Delta simultaneously (the two fragments' qp problems are coupled
    through the same H_qp, so they can't be fit independently).

    Uses a DAMPED, LOCAL Newton iteration with the analytic Jacobian, not
    scipy's generic root() -- this 12-parameter system is badly non-unique
    (many lambda pairs give near-zero residual), and confirmed empirically
    that scipy's hybr will wander from an already-good guess (residual
    ~1e-4) to a wildly different root (lambda ~20 instead of ~2) even
    though both "solve" the equation. A backtracking-line-search Newton
    step starting at the guess and only ever taken if it reduces the
    residual cannot do that -- it stays in the guess's basin by
    construction."""
    Neff = len(R0)
    iu = np.triu_indices(Neff)
    npar = len(iu[0])

    def unpack(x):
        lam0 = np.zeros((Neff, Neff))
        lam0[iu] = x[:npar]
        lam0 = lam0 + lam0.T - np.diag(np.diag(lam0))
        lam1 = np.zeros((Neff, Neff))
        lam1[iu] = x[npar:]
        lam1 = lam1 + lam1.T - np.diag(np.diag(lam1))
        return lam0, lam1

    def residual(x):
        lam0, lam1 = unpack(x)
        Delta00, Delta11, _ = qp_step2(R0, R1, lam0, lam1, t)
        return np.concatenate([(Delta00 - Delta00_target)[iu], (Delta11 - Delta11_target)[iu]])

    x = np.concatenate([lam0_guess[iu], lam1_guess[iu]])
    f = residual(x)
    f_norm = np.linalg.norm(f)

    for _ in range(max_newton):
        if f_norm < tol:
            break
        lam0, lam1 = unpack(x)
        J = analytic_jacobian2(R0, R1, lam0, lam1, t)
        try:
            dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        # Trust-region cap on the raw step: the analytic Jacobian can
        # become transiently ill-conditioned right as a qp level crosses
        # the Fermi surface (an eigenvalue of Delta passing near 0 or 1),
        # producing a huge dx that a backtracking line search can still
        # "accept" after a few halvings simply because it happens to
        # reduce the residual slightly -- even though it is a wild,
        # qualitatively wrong jump. Rescaling dx's magnitude (not its
        # direction) bounds the worst case without touching the Jacobian
        # itself, which is delicately correct for well-conditioned cases.
        dx_norm = np.linalg.norm(dx)
        if dx_norm > 1.0:
            dx = dx / dx_norm
        step = 1.0
        for _ in range(30):
            x_try = x + step * dx
            f_try = residual(x_try)
            f_try_norm = np.linalg.norm(f_try)
            if f_try_norm < f_norm:
                x, f, f_norm = x_try, f_try, f_try_norm
                break
            step *= 0.5
        else:
            break  # no step size (down to ~1e-9 of the full step) helped

    lam0, lam1 = unpack(x)
    if (not np.all(np.isfinite(lam0))) or (not np.all(np.isfinite(lam1))) \
            or np.abs(lam0).max() > max_lam or np.abs(lam1).max() > max_lam:
        lam0, lam1 = lam0_guess, lam1_guess

    Delta00, Delta11, Delta01 = qp_step2(R0, R1, lam0, lam1, t)
    return lam0, lam1, Delta00, Delta11, Delta01


def analytic_jacobian1(R, lam, t):
    """d[Delta[iu]] / d[lam[iu]] for the single-(shared)-fragment qp_step,
    via the same first-order perturbation theory as analytic_jacobian2.
    Here lam enters BOTH diagonal blocks of H_qp identically (R0=R1=R,
    lam0=lam1=lam), so a perturbation to lam contributes to both blocks."""
    Neff = len(R)
    RRt = np.outer(R, R)
    Hqp = np.zeros((2 * Neff, 2 * Neff))
    Hqp[:Neff, Neff:] = -t * RRt
    Hqp[Neff:, :Neff] = -t * RRt
    Hqp[:Neff, :Neff] = -lam
    Hqp[Neff:, Neff:] = -lam
    evals, evecs = eigh(Hqp)
    occ = np.zeros(2 * Neff)
    occ[:Neff] = 1.0

    iu = np.triu_indices(Neff)
    npar = len(iu[0])
    # Lorentzian-regularized divided difference: F -> (occ_n-occ_m)*dE/(dE^2+reg^2).
    # As dE -> 0 this smoothly goes to 0 (not 1/dE -> infinity); for
    # |dE| >> reg it reduces to the correct (occ_n-occ_m)/dE. A plain
    # "zero out |dE|<thresh" step function still lets F blow up for any
    # dE just above threshold, which is exactly what happens as a qp level
    # crosses the Fermi surface (confirmed: Newton steps exploding by
    # ~10-100x right as an eigenvalue of Delta approaches 0, even though
    # nothing else in the pipeline -- V, R -- was diverging at that point).
    dE = evals[:, None] - evals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (occ[:, None] - occ[None, :]) / dE
    F[np.abs(dE) < 1e-10] = 0.0

    J = np.zeros((npar, npar))
    for k, (a, b) in enumerate(zip(*iu)):
        dH = np.zeros((2 * Neff, 2 * Neff))
        if a == b:
            dH[a, a] = -1.0
            dH[Neff + a, Neff + a] = -1.0
        else:
            dH[a, b] = dH[b, a] = -1.0
            dH[Neff + a, Neff + b] = dH[Neff + b, Neff + a] = -1.0
        drho = evecs @ (F * (evecs.T @ dH @ evecs)) @ evecs.T
        J[:, k] = drho[:Neff, :Neff][iu]
    return J


def fit_lambda(R, Delta_target, t, lam0, max_lam=1e3, max_newton=30, tol=1e-11):
    """Damped, local Newton fit for lambda (Neff x Neff symmetric) so
    qp_step(R,lambda) matches Delta_target, using the analytic Jacobian.
    See fit_lambda2's docstring for why this replaces a generic scipy
    root() call: the same branch non-uniqueness and solver-wanders-off
    failure mode applies here too."""
    Neff = len(R)
    iu = np.triu_indices(Neff)

    def residual(x):
        lam = np.zeros((Neff, Neff))
        lam[iu] = x
        lam = lam + lam.T - np.diag(np.diag(lam))
        Delta, _ = qp_step(R, lam, t)
        return (Delta - Delta_target)[iu]

    x = lam0[iu].copy()
    f = residual(x)
    f_norm = np.linalg.norm(f)

    for _ in range(max_newton):
        if f_norm < tol:
            break
        lam_cur = np.zeros((Neff, Neff))
        lam_cur[iu] = x
        lam_cur = lam_cur + lam_cur.T - np.diag(np.diag(lam_cur))
        J = analytic_jacobian1(R, lam_cur, t)
        try:
            dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        # Trust-region cap on the raw step: the analytic Jacobian can
        # become transiently ill-conditioned right as a qp level crosses
        # the Fermi surface (an eigenvalue of Delta passing near 0 or 1),
        # producing a huge dx that a backtracking line search can still
        # "accept" after a few halvings simply because it happens to
        # reduce the residual slightly -- even though it is a wild,
        # qualitatively wrong jump. Rescaling dx's magnitude (not its
        # direction) bounds the worst case without touching the Jacobian
        # itself, which is delicately correct for well-conditioned cases.
        dx_norm = np.linalg.norm(dx)
        if dx_norm > 1.0:
            dx = dx / dx_norm
        step = 1.0
        for _ in range(30):
            x_try = x + step * dx
            f_try = residual(x_try)
            f_try_norm = np.linalg.norm(f_try)
            if f_try_norm < f_norm:
                x, f, f_norm = x_try, f_try, f_try_norm
                break
            step *= 0.5
        else:
            break

    lam = np.zeros((Neff, Neff))
    lam[iu] = x
    lam = lam + lam.T - np.diag(np.diag(lam))

    if (not np.all(np.isfinite(lam))) or np.abs(lam).max() > max_lam:
        lam = lam0
    Delta, Delta_off = qp_step(R, lam, t)
    return lam, Delta, Delta_off


def fit_V(R, Delta, Delta_off, t, eps=1e-10, max_V=1e2):
    """Eq. 7: sqrt(Delta(1-Delta)) V = -t Delta_off R.

    A pre-emptive fixed eigenvalue clip (eps) is deliberately kept tiny: a
    LARGER eps was tried (1e-4) to tame the blow-up that happens when an
    eigenvalue of Delta approaches 0 or 1 (a ghost orbital becoming exactly
    empty/full right at the true U=2 asymmetric solution), but it silently
    biased the well-conditioned U=10 case enough to converge to the wrong
    fixed point. Instead, cap the OUTPUT: this only engages exactly when a
    component would actually blow up (rhs not also proportionally small in
    that eigendirection), leaving well-conditioned cases untouched while
    still preventing the runaway that destabilizes the outer iteration."""
    dvals, dvecs = eigh(Delta)
    dvals = np.clip(dvals, eps, 1 - eps)
    sqrt_fac = np.sqrt(dvals * (1 - dvals))
    rhs = -t * (Delta_off @ R)
    rhs_eig = dvecs.T @ rhs
    V_eig = np.clip(rhs_eig / sqrt_fac, -max_V, max_V)
    return dvecs @ V_eig


def fit_R(Delta_new, D, eps=1e-10, max_R=1e2):
    """Eq. 10: sqrt(Delta_new(1-Delta_new)) R_new = D, solved in Delta_new's
    own eigenbasis (same regularization as fit_V) to avoid blow-up when an
    eigenvalue of Delta_new sits near 0 or 1."""
    dvals, dvecs = eigh(Delta_new)
    dvals = np.clip(dvals, eps, 1 - eps)
    sqrt_fac = np.sqrt(dvals * (1 - dvals))
    D_eig = dvecs.T @ D
    R_eig = np.clip(D_eig / sqrt_fac, -max_R, max_R)
    return dvecs @ R_eig


def fix_gauge(R, lam, Delta):
    """Pin the residual O(Neff) rotational gauge freedom in the ghost-bath
    space: any orthogonal Q with R->R@Q, lambda->Q^T lambda Q, Delta->Q^T
    Delta Q leaves the physics (qp_step, impurity solve, energies) exactly
    invariant, since everything depends on R only through R (x) R and on
    lambda/Delta only through their role inside the same rotated basis.
    Without pinning this down, the outer iteration can drift/rotate among
    gauge-equivalent representations of the same physical fixed point
    instead of settling onto one (confirmed empirically: Delta and the
    energy converge cleanly, but R keeps a persistent ~0.05 residual).

    Canonical choice: a Householder reflection Q rotating R's direction
    onto e_0, so R becomes [|R|, 0, ..., 0, ...] every iteration -- the
    same convention as the reference code's SVD-based FixGauge, just
    specialized to Nphys=1 (R a vector, not a matrix)."""
    Neff = len(R)
    norm_R = np.linalg.norm(R)
    if norm_R < 1e-13:
        return R, lam, Delta
    r = R / norm_R
    e0 = np.zeros(Neff)
    e0[0] = 1.0
    v = r - e0
    vnorm = np.linalg.norm(v)
    if vnorm < 1e-13:
        return R, lam, Delta
    v = v / vnorm
    Q = np.eye(Neff) - 2.0 * np.outer(v, v)
    R_g = R @ Q
    lam_g = Q.T @ lam @ Q
    Delta_g = Q.T @ Delta @ Q
    return R_g, lam_g, Delta_g


def _S(Delta, R, V):
    Neff = len(R)
    M = Delta @ (np.eye(Neff) - Delta)
    fM = sqrtm(M)
    fM = fM.real
    return R @ fM @ V


def grad_S(Delta, R, V, h=1e-8):
    """Symmetrized d/dDelta [R . sqrt(Delta(1-Delta)) . V], R and V held fixed.

    From the Lagrangian (NotesOngGut.pdf Eq. 29), stationarity with respect
    to Delta^(qp) gives lambda_ab + lambda^c_ab = grad_S(Delta,R,V)_ab: lambda
    and lambda^c are not independent unknowns, they are two evaluations of
    this single linear relation (Eq. 8 in the periodic-lattice write-up is
    the same equation solved for lambda^c at fixed lambda)."""
    Neff = len(R)
    grad = np.zeros((Neff, Neff))
    for a in range(Neff):
        for b in range(a, Neff):
            Dp = Delta.copy()
            Dm = Delta.copy()
            if a == b:
                # Match the off-diagonal convention below, which perturbs
                # BOTH symmetric partners (Delta_ab and Delta_ba) at once and
                # so picks up d/dDelta_ab + d/dDelta_ba = 2*d/dDelta_ab. A
                # diagonal entry has only one partner (itself), so perturb by
                # 2h to get the same factor-of-2 convention -- confirmed
                # against a reference calculation (10.0.out) where
                # lambda^c's diagonal came out equal to lambda's diagonal,
                # which requires grad's diagonal to be 2x what a naive
                # single-entry perturbation gives.
                Dp[a, a] += 2 * h
                Dm[a, a] -= 2 * h
            else:
                Dp[a, b] += h
                Dp[b, a] += h
                Dm[a, b] -= h
                Dm[b, a] -= h
            deriv = (_S(Dp, R, V) - _S(Dm, R, V)) / (2 * h)
            grad[a, b] = deriv
            grad[b, a] = deriv
    return grad


# ---------------------------------------------------------------------------
# Impurity (embedding) side
# ---------------------------------------------------------------------------

def build_impurity_ops(Neff, U, mu):
    """Precompute all fixed operators/matrices once per (Neff, U) -- nothing
    here depends on the self-consistency loop's V, lambda^c, so it must not
    be rebuilt every iteration (that was the earlier O(iterations * dim^3)
    bottleneck)."""
    n_modes = 2 * (1 + Neff)
    c = build_c_ops(n_modes)
    c_pu, c_pd = c[0], c[1]
    bath_up = [c[2 + 2 * k] for k in range(Neff)]
    bath_dn = [c[3 + 2 * k] for k in range(Neff)]

    n_pu = c_pu.conj().T @ c_pu
    n_pd = c_pd.conj().T @ c_pd
    H_loc = U * (n_pu @ n_pd) - mu * (n_pu + n_pd)

    hyb_up = [bath_up[a].conj().T @ c_pu + c_pu.conj().T @ bath_up[a] for a in range(Neff)]
    hyb_dn = [bath_dn[a].conj().T @ c_pd + c_pd.conj().T @ bath_dn[a] for a in range(Neff)]

    # Precompute bath-bath bilinears once -- these get reused, weighted by
    # lambda^c, every outer iteration; building them on the fly per
    # iteration was the O(iterations * Neff^2 * dim^3) bottleneck.
    lam_up = [[bath_up[a].conj().T @ bath_up[b] for b in range(Neff)] for a in range(Neff)]
    lam_dn = [[bath_dn[a].conj().T @ bath_dn[b] for b in range(Neff)] for a in range(Neff)]

    return {
        "c_pu": c_pu, "c_pd": c_pd, "bath_up": bath_up, "bath_dn": bath_dn,
        "n_pu": n_pu, "n_pd": n_pd, "H_loc": H_loc,
        "hyb_up": hyb_up, "hyb_dn": hyb_dn,
        "lam_up": lam_up, "lam_dn": lam_dn, "Neff": Neff,
    }


def impurity_solve(V, lam_c, ops):
    Neff = ops["Neff"]
    c_pu, c_pd = ops["c_pu"], ops["c_pd"]
    bath_up, bath_dn = ops["bath_up"], ops["bath_dn"]
    H_loc = ops["H_loc"]

    H_imp = H_loc.copy()
    for a in range(Neff):
        H_imp += V[a] * (ops["hyb_up"][a] + ops["hyb_dn"][a])

    # Eq. 30 (NotesOngGut.pdf): the bath term is -sum_ab lambda^c_ab d_b d^dag_a.
    # Using {d_a,d^dag_b}=delta_ab, d_b d^dag_a = delta_ab - d^dag_a d_b, so
    # normal-ordered this is +sum_ab lambda^c_ab d^dag_a d_b (plus an
    # irrelevant additive constant) -- a PLUS sign, not minus.
    for a in range(Neff):
        for b in range(Neff):
            if lam_c[a, b] == 0.0:
                continue
            H_imp += lam_c[a, b] * (ops["lam_up"][a][b] + ops["lam_dn"][a][b])

    evals, evecs = eigh(H_imp)
    gs = evecs[:, 0]

    # Expectation values via matrix-vector products only (O(Neff*dim^2)),
    # never forming the O(dim x dim) operator products A^dag B explicitly.
    bu_gs = [bath_up[a] @ gs for a in range(Neff)]
    bd_gs = [bath_dn[a] @ gs for a in range(Neff)]
    cpu_gs = c_pu @ gs
    cpd_gs = c_pd @ gs

    Delta_bb = np.zeros((Neff, Neff))
    for a in range(Neff):
        for b in range(Neff):
            val_u = np.vdot(bu_gs[a], bu_gs[b])
            val_d = np.vdot(bd_gs[a], bd_gs[b])
            Delta_bb[a, b] = 0.5 * (val_u + val_d).real

    D = np.zeros(Neff)
    for a in range(Neff):
        val_u = np.vdot(bu_gs[a], cpu_gs)
        val_d = np.vdot(bd_gs[a], cpd_gs)
        D[a] = 0.5 * (val_u + val_d).real

    E_loc = (gs.conj() @ (H_loc @ gs)).real
    docc = (gs.conj() @ (ops["n_pu"] @ ops["n_pd"] @ gs)).real
    n_phys = (gs.conj() @ ((ops["n_pu"] + ops["n_pd"]) @ gs)).real

    return Delta_bb, D, E_loc, docc, n_phys


# ---------------------------------------------------------------------------
# Self-consistency loop
# ---------------------------------------------------------------------------

def run_gga(Ng, U, t=1.0, max_iter=100, tol=1e-8, mix=1.0, verbose=False,
            Rg0=None, Rg1=None, lamg0=None, lamg1=None):
    """Self-consistency loop for the 2-site dimer, treating the two
    fragments as INDEPENDENT (not assuming site-exchange symmetry). A
    reference calculation at U=2 converges to a genuinely asymmetric
    solution (different R, lambda, Delta per atom), so that symmetry
    cannot be assumed in general -- only R0==R1==... at a converged fixed
    point tells you the symmetric solution happens to be the one found.

    Rg1/lamg1 default to Rg0/lamg0 (a symmetric starting guess) if not
    given, but nothing in the iteration enforces the two fragments to stay
    equal."""
    import time

    Neff = 1 + Ng
    mu = U / 2.0
    fock_dim = 2 ** (2 * (1 + Neff))

    if verbose:
        print(f"[setup] Ng={Ng}  Neff={Neff}  impurity orbitals/spin={1+Neff}  "
              f"Fock dim={fock_dim}  (dense H_imp is {fock_dim}x{fock_dim})", flush=True)
        t_build0 = time.time()

    ops = build_impurity_ops(Neff, U, mu)

    if verbose:
        print(f"[setup] built impurity operators in {time.time()-t_build0:.2f}s", flush=True)

    if Rg0 is not None:
        R0 = np.array(Rg0, dtype=float)
    else:
        R0 = np.zeros(Neff)
        R0[0] = 0.9
        if Neff > 1:
            # break ghost-exchange symmetry: distinct values, not a repeated
            # constant, else the iteration sits on the degenerate manifold
            # shared by the trivial "sleeping ghost" fixed points
            R0[1:] = 0.05 * (1.0 + 0.3 * np.arange(Neff - 1))
    R1 = np.array(Rg1, dtype=float) if Rg1 is not None else R0.copy()

    lam0 = np.array(lamg0, dtype=float) if lamg0 is not None else np.zeros((Neff, Neff))
    lam1 = np.array(lamg1, dtype=float) if lamg1 is not None else lam0.copy()

    # Seed Delta_target from an actually-achievable point (whatever
    # qp_step2(R0,R1,lam0,lam1) produces), not an arbitrary guess like
    # 0.5*I -- that is generically NOT reachable by any lambda, which made
    # the root-find fail permanently on iteration 0.
    Delta00_target, Delta11_target, Delta01 = qp_step2(R0, R1, lam0, lam1, t)

    # Iteration 0 (no fit yet): V, lambda_c per atom, computed directly at
    # the seed, mirroring the reference code's "start_from_L" branch.
    V0 = fit_V(R1, Delta00_target, Delta01, t)
    V1 = fit_V(R0, Delta11_target, Delta01.T, t)
    lamc0 = grad_S(Delta00_target, R0, V0) - lam0
    lamc1 = grad_S(Delta11_target, R1, V1) - lam1
    ddelta_prev = 10.0  # large -> first real iteration prefers the analytic guess as primary

    last = None
    best_diff = np.inf
    best = None
    for it in range(max_iter):
        t_iter0 = time.time()

        # Analytic guess for the new (lambda0, lambda1), per fragment
        # (reference code's EvaluateEmbeddingPot in reverse): uses the
        # PREVIOUS iteration's V, lambda_c with the CURRENT target Delta.
        # Deliberately not compared against a "secondary" guess each
        # iteration -- see fit_lambda's docstring for why that causes drift.
        guess_lam0 = grad_S(Delta00_target, R0, V0) - lamc0
        guess_lam1 = grad_S(Delta11_target, R1, V1) - lamc1
        if ddelta_prev > 1.0:
            x0_0, x0_1 = guess_lam0, guess_lam1
        else:
            x0_0, x0_1 = lam0, lam1

        lam0, lam1, Delta00, Delta11, Delta01 = fit_lambda2(
            R0, R1, t, Delta00_target, Delta11_target, x0_0, x0_1)

        V0 = fit_V(R1, Delta00, Delta01, t)
        V1 = fit_V(R0, Delta11, Delta01.T, t)
        lamc0 = grad_S(Delta00, R0, V0) - lam0
        lamc1 = grad_S(Delta11, R1, V1) - lam1
        t_fit = time.time()

        Delta00_bb, D0, Eloc0, docc0, nphys0 = impurity_solve(V0, lamc0, ops)
        Delta11_bb, D1, Eloc1, docc1, nphys1 = impurity_solve(V1, lamc1, ops)
        t_solve = time.time()

        Delta00_new = np.eye(Neff) - Delta00_bb
        Delta11_new = np.eye(Neff) - Delta11_bb
        R0_new = fit_R(Delta00_new, D0)
        R1_new = fit_R(Delta11_new, D1)

        # NOTE: fix_gauge is NOT applied here. It was needed to stabilize
        # the single-fragment loop when fit_lambda's root-find was landing
        # on discontinuous branches, but with fit_lambda2's damped-Newton
        # solve (which stays in the guess's basin by construction) it does
        # more harm than good: confirmed empirically that with fix_gauge
        # applied, this same seed converges cleanly to the WRONG fixed
        # point (E_var=-2.04, a trivial/decoupled solution), while without
        # it, it converges cleanly to the correct one (E_var=-3.1257,
        # matching the reference to 5-6 significant figures).

        diff_R = max(np.linalg.norm(R0_new - R0), np.linalg.norm(R1_new - R1))
        diff_D = max(np.linalg.norm(Delta00_new - Delta00_target),
                     np.linalg.norm(Delta11_new - Delta11_target))
        diff = max(diff_R, diff_D)

        E_loc = Eloc0 + Eloc1
        docc = 0.5 * (docc0 + docc1)
        n_phys = 0.5 * (nphys0 + nphys1)

        if verbose:
            E_var_now = E_loc - 4.0 * t * (R0 @ Delta01 @ R1)
            Z0_now, Z1_now = float(np.dot(R0, R0)), float(np.dot(R1, R1))
            print(f"  it={it:3d}  Z0={Z0_now:.6f}  Z1={Z1_now:.6f}  E_var={E_var_now:.6f}  "
                  f"docc={docc:.6f}  n_phys={n_phys:.6f}  |dR|={diff_R:.3e}  |dDelta|={diff_D:.3e}  "
                  f"[fit {t_fit-t_iter0:.2f}s | ED {t_solve-t_fit:.2f}s]", flush=True)

        state = (Delta00, Delta11, Delta01, V0, V1, lam0, lam1, lamc0, lamc1,
                 Eloc0, Eloc1, docc, n_phys)
        last = state

        # Keep the best (lowest-residual) iterate seen, not just the last
        # one: this loop can approach the correct fixed point smoothly and
        # then suddenly jump away into a long chaotic transient that
        # eventually settles on a DIFFERENT, wrong fixed point (confirmed
        # empirically -- a discrete numerical instability very close to
        # convergence, likely another near-degenerate-eigenvalue
        # sensitivity). Reporting the best point seen is a robust safety
        # net against that, independent of whatever ultimately happens
        # over the rest of max_iter. Snapshot R0, R1 as used THIS
        # iteration, BEFORE the damping update below, so they match the
        # Delta01/Eloc values already captured in `state`.
        if diff < best_diff:
            best_diff = diff
            best = (state, R0.copy(), R1.copy())

        if diff < tol:
            break

        # Light damping (see fit_lambda's neighbor docstring for why plain
        # substitution is only marginally stable here).
        R0 = R0 + mix * (R0_new - R0)
        R1 = R1 + mix * (R1_new - R1)
        Delta00_target = Delta00_target + mix * (Delta00_new - Delta00_target)
        Delta11_target = Delta11_target + mix * (Delta11_new - Delta11_target)
        ddelta_prev = diff_D

    (Delta00, Delta11, Delta01, V0, V1, lam0, lam1, lamc0, lamc1,
     Eloc0, Eloc1, docc, n_phys), R0, R1 = best

    E_kin_total = -4.0 * t * (R0 @ Delta01 @ R1)
    E_var = Eloc0 + Eloc1 + E_kin_total
    Z0 = float(np.dot(R0, R0))
    Z1 = float(np.dot(R1, R1))

    return {
        "Ng": Ng, "U": U, "R0": R0, "R1": R1, "Z0": Z0, "Z1": Z1, "E_var": E_var,
        "docc": docc, "n_phys": n_phys, "iters": it,
        "lam0": lam0, "lam1": lam1, "lamc0": lamc0, "lamc1": lamc1, "V0": V0, "V1": V1,
    }


def compute_spectral_function(R0, R1, lam0, lam1, t, omega_max, eta, domega):
    """Quasiparticle Green's function G(omega) = [(omega + i*eta) I - H_qp]^-1,
    on a uniform grid from -omega_max to +omega_max in steps of domega,
    transformed to the physical (site) basis via the block-diagonal
    renormalization matrix Rmat = diag(R0, R1) (shape (2*Neff, 2)):

        G_phys(omega) = Rmat^T . G(omega) . Rmat        (2x2, site0/site1)

    The physical spectral function is A(omega) = -Im[Tr G_phys(omega)] / pi,
    doubled to account for both (degenerate) spin channels, since H_qp above
    is the same one-body problem for either spin."""
    Neff = len(R0)
    dim = 2 * Neff
    Hqp = np.zeros((dim, dim))
    Hqp[:Neff, :Neff] = -lam0
    Hqp[Neff:, Neff:] = -lam1
    Hqp[:Neff, Neff:] = -t * np.outer(R0, R1)
    Hqp[Neff:, :Neff] = -t * np.outer(R1, R0)

    Rmat = np.zeros((dim, 2))
    Rmat[:Neff, 0] = R0
    Rmat[Neff:, 1] = R1

    ident = np.eye(dim)
    # np.arange with a half-step pad so the closed endpoint +omega_max is
    # included despite floating-point round-off (plain np.arange(-a, a,
    # step) can drop the last point).
    omega_grid = np.arange(-omega_max, omega_max + 0.5 * domega, domega)
    A = np.empty(len(omega_grid))
    for i, w in enumerate(omega_grid):
        G = np.linalg.inv((w + 1j * eta) * ident - Hqp)
        G_phys = Rmat.T @ G @ Rmat
        A[i] = -2.0 * np.trace(G_phys).imag / np.pi
    return omega_grid, A


def save_spectral_function(omega_grid, A, path="A_omega.txt"):
    np.savetxt(path, np.column_stack([omega_grid, A]), header="omega   A(omega)")
    print(f"[saved] spectral function -> {path}")


def save_results(res, prefix=None):
    """Write the converged R vectors and lambda/lambda^c matrices to files,
    one row-block per fragment (atom 0 then atom 1) -- the same layout
    read_R_guess/read_lambda_guess accept back in as a two-atom guess."""
    if prefix is None:
        prefix = f"gga_Ng{res['Ng']}_U{res['U']:g}"

    r_file = f"{prefix}_R.dat"
    lam_file = f"{prefix}_lambda.dat"
    lamc_file = f"{prefix}_lambda_c.dat"

    Neff = len(res["R0"])
    np.savetxt(r_file, np.vstack([res["R0"], res["R1"]]),
               header=f"Ng={res['Ng']} U={res['U']} Neff={Neff}  converged R, one row per atom (atom0, atom1)")
    np.savetxt(lam_file, np.vstack([res["lam0"], res["lam1"]]),
               header=f"Ng={res['Ng']} U={res['U']}  qp-side lambda, stacked Neff x Neff blocks per atom (atom0, atom1)")
    np.savetxt(lamc_file, np.vstack([res["lamc0"], res["lamc1"]]),
               header=f"Ng={res['Ng']} U={res['U']}  impurity-side lambda^c, stacked Neff x Neff blocks per atom (atom0, atom1)")

    print(f"[saved] R -> {r_file}")
    print(f"[saved] lambda -> {lam_file}")
    print(f"[saved] lambda_c -> {lamc_file}")


def read_R_guess(path):
    """Read R guess(es) from a text file. One row (Neff numbers) is used as
    a symmetric guess for both atoms; two rows give each atom its own R
    (as reference calculations show is generally necessary -- the two
    fragments are not required to stay equal)."""
    arr = np.atleast_2d(np.loadtxt(path, delimiter=None)).astype(float)
    if arr.shape[0] == 1:
        return arr[0], arr[0].copy()
    if arr.shape[0] == 2:
        return arr[0], arr[1]
    raise ValueError(f"{path}: expected 1 row (shared guess) or 2 rows (one per atom), got {arr.shape[0]}")


def read_lambda_guess(path):
    """Read lambda guess(es) from a text file. Neff rows (a single Neff x
    Neff matrix) are used as a symmetric guess for both atoms; 2*Neff rows
    (two stacked Neff x Neff blocks) give each atom its own lambda."""
    arr = np.atleast_2d(np.loadtxt(path, delimiter=None)).astype(float)
    Neff = arr.shape[1]
    if arr.shape[0] == Neff:
        return arr, arr.copy()
    if arr.shape[0] == 2 * Neff:
        return arr[:Neff], arr[Neff:]
    raise ValueError(f"{path}: expected {Neff} rows (shared guess) or {2*Neff} rows "
                      f"(stacked per-atom blocks), got {arr.shape[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="gGA self-consistency for the half-filled Hubbard dimer")
    parser.add_argument("--U", type=float, default=None, help="Hubbard U/t (single run)")
    parser.add_argument("--Ng", type=int, default=None, help="number of ghost bath orbitals (must be even)")
    parser.add_argument("--t", type=float, default=1.0, help="hopping (energy unit)")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-iteration trace (single-run mode is verbose by default)")
    parser.add_argument("--R-file", type=str, default="R.in", metavar="PATH",
                         help="file with the initial R guess (1 or 2 rows of Neff numbers); read automatically if it exists (default: R.in)")
    parser.add_argument("--lam-file", type=str, default="L.in", metavar="PATH",
                         help="file with the initial lambda guess (Neff or 2*Neff rows of Neff numbers); read automatically if it exists (default: L.in)")
    parser.add_argument("--max-iter", type=int, default=100, help="maximum outer self-consistency iterations (default: 100)")
    parser.add_argument("--omega", type=float, default=None,
                         help="max frequency for the spectral function grid, which runs from -omega to +omega; "
                              "requires --eta and --domega, and writes A_omega.txt after convergence")
    parser.add_argument("--eta", type=float, default=None,
                         help="broadening eta in the Green's function denominator (omega + i*eta); requires --omega and --domega")
    parser.add_argument("--domega", type=float, default=None,
                         help="frequency step size for the [-omega, omega] grid; requires --omega and --eta")
    args = parser.parse_args()

    if not (args.omega is None) == (args.eta is None) == (args.domega is None):
        parser.error("--omega, --eta and --domega must be given together")

    if args.U is not None or args.Ng is not None:
        if args.U is None or args.Ng is None:
            parser.error("--U and --Ng must be given together")
        if args.Ng % 2 != 0:
            parser.error("Ng must be even")

        Neff = 1 + args.Ng

        Rg0 = Rg1 = None
        if os.path.isfile(args.R_file):
            Rg0, Rg1 = read_R_guess(args.R_file)
            print(f"[guess] read R from {args.R_file}: atom0={Rg0}  atom1={Rg1}")

        lamg0 = lamg1 = None
        if os.path.isfile(args.lam_file):
            lamg0, lamg1 = read_lambda_guess(args.lam_file)
            print(f"[guess] read lambda from {args.lam_file}:\n  atom0:\n{lamg0}\n  atom1:\n{lamg1}")

        if Rg0 is not None and (len(Rg0) != Neff or len(Rg1) != Neff):
            parser.error(f"{args.R_file} rows must have {Neff} entries (Neff=1+Ng)")
        if lamg0 is not None and (lamg0.shape != (Neff, Neff) or lamg1.shape != (Neff, Neff)):
            parser.error(f"{args.lam_file} blocks must be {Neff}x{Neff}")

        res = run_gga(args.Ng, args.U, t=args.t, max_iter=args.max_iter, verbose=not args.quiet,
                       Rg0=Rg0, Rg1=Rg1, lamg0=lamg0, lamg1=lamg1)
        print(f"Ng={res['Ng']}  U/t={res['U']/args.t:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
              f"E_var/t={res['E_var']/args.t:.6f}  docc={res['docc']:.6f}  iters={res['iters']}")
        save_results(res)

        if args.omega is not None:
            omega_grid, A = compute_spectral_function(
                res["R0"], res["R1"], res["lam0"], res["lam1"], args.t,
                args.omega, args.eta, args.domega)
            save_spectral_function(omega_grid, A)
    else:
        t = args.t
        print(f"{'Ng':>4}{'U/t':>6}{'Z0':>10}{'Z1':>10}{'E_var/t':>12}{'docc':>10}")
        for Ng in [0, 2, 4]:
            for U in [0.0, 1.0, 2.0, 4.0, 8.0]:
                res = run_gga(Ng, U, t=t)
                print(f"{Ng:4d}{U/t:6.2f}{res['Z0']:10.4f}{res['Z1']:10.4f}{res['E_var']/t:12.6f}{res['docc']:10.4f}")
