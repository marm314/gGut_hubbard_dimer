"""Check that a converged gGA solution is a stationary point of the Lagrangian.

The self-consistency equations of gga_dimer.py are the Euler-Lagrange equations
of

  L = E_kin[R, rho] + sum_I { <Phi_I|Hloc|Phi_I>
        + 4 V_I . (D[Phi_I] - S(Delta_I) R_I)
        + 2 Tr[ lamc_I (Delta_I + Dbb[Phi_I] - <Phi_I|Phi_I>) ]
        + 2 Tr[ lam_I  (Delta_I - rho^II) ] }

with E_kin = -2t (R0 rho01 R1 + R1 rho10 R0)  (2 = spin sum, both bonds),
rho = C C^T the 1-RDM (per spin) of the Slater determinant Psi built from the
Neff occupied orbitals C of 2*Neff, Phi_I the impurity states (real vectors in
the N_up = N_dn sector, unit norm enforced by a multiplier), D, Dbb the
spin-averaged bath-impurity and bath-bath blocks, and S the shifted matrix
square root sqrt(Delta(1-Delta)+shift). Spin factors are chosen so that the
stationarity conditions are exactly the equations of gga_dimer.py:

  dL/dV      = 4 (D - S R)                       (R update, Eq. Rupdate)
  dL/dlamc   = 2 (Delta + Dbb - 1)               (bath density = 1 - Delta)
  dL/dlam    = 2 (Delta - rho^II)                (qp density = Delta)
  dL/dR^0    = -4t rho01 R^1 - 4 S(Delta0) V0    (Eq. V; R^1 analogous)
  dL/dDelta  = 2 (lam + lamc) - 2 G(Delta, V R)  (Eq. lamc)
  dL/dPhi    -> H_imp Phi = E Phi   (tangent gradient 2 (H - <H>) Phi)
  dL/dPsi    -> [H_qp, rho] = 0     (rotations occupied -> virtual)

Every block is evaluated (a) analytically and (b) by central finite
differences of L itself, at the converged point of run_gga and at a
deliberately perturbed point (which must NOT be stationary).
"""

import sys

import numpy as np
from scipy.linalg import expm

import gga_dimer as g


def S_of(Delta, shift):
    d, W = np.linalg.eigh(Delta)
    return (W * np.sqrt(d * (1.0 - d) + shift)) @ W.T


def sector_vec(phi, ops):
    full = np.zeros(ops["H_loc"].shape[0])
    full[ops["sector"]] = phi
    return full


def imp_terms(phi, ops):
    """Unnormalized expectation values of the impurity state phi (sector coeffs)."""
    Neff = ops["Neff"]
    x = sector_vec(phi, ops)
    bu = [ops["bath_up"][a] @ x for a in range(Neff)]
    bd = [ops["bath_dn"][a] @ x for a in range(Neff)]
    cu, cd = ops["c_pu"] @ x, ops["c_pd"] @ x
    Eloc = x @ (ops["H_loc"] @ x)
    D = np.array([0.5 * (bu[a] @ cu + bd[a] @ cd) for a in range(Neff)])
    Dbb = np.array([[0.5 * (bu[a] @ bu[b] + bd[a] @ bd[b]) for b in range(Neff)] for a in range(Neff)])
    return Eloc, D, Dbb, x @ x


def imp_sector_H(V, lamc, ops):
    """Sector block of H_imp exactly as built in gga_dimer.impurity_solve."""
    Neff = ops["Neff"]
    H = ops["H_loc"]
    for a in range(Neff):
        H = H + V[a] * (ops["hyb_up"][a] + ops["hyb_dn"][a])
        for b in range(Neff):
            H = H + lamc[a, b] * (ops["lam_up"][a][b] + ops["lam_dn"][a][b])
    sec = ops["sector"]
    return H.tocsr()[sec][:, sec].toarray()


def Lfun(p, ops, t, shift):
    """Lagrangian; p = dict(R, Delta, V, lam, lamc, C, phi)."""
    Neff = ops["Neff"]
    rho = p["C"] @ p["C"].T
    R0, R1 = p["R"]
    rho01, rho10 = rho[:Neff, Neff:], rho[Neff:, :Neff]
    L = -2.0 * t * (R0 @ rho01 @ R1 + R1 @ rho10 @ R0)
    for I in range(2):
        Eloc, D, Dbb, nrm = imp_terms(p["phi"][I], ops)
        S = S_of(p["Delta"][I], shift)
        rhoII = rho[I * Neff:(I + 1) * Neff, I * Neff:(I + 1) * Neff]
        L += Eloc + 4.0 * p["V"][I] @ (D - S @ p["R"][I])
        L += 2.0 * np.trace(p["lamc"][I] @ (p["Delta"][I] + Dbb - nrm * np.eye(Neff)))
        L += 2.0 * np.trace(p["lam"][I] @ (p["Delta"][I] - rhoII))
    return L


def sym_dirs(N):
    out = []
    for i in range(N):
        for j in range(i, N):
            d = np.zeros((N, N))
            d[i, j] = 1.0
            d[j, i] = 1.0
            out.append(d)
    return out


def analysis(p, ops, t, shift, hs=(1e-6, 1e-8, 1e-10, 1e-12)):
    """Return {block: (max|numeric|, max|analytic|)} at the point p."""
    Neff = ops["Neff"]
    N2 = 2 * Neff
    rho = p["C"] @ p["C"].T
    Hqp = np.zeros((N2, N2))
    for I in range(2):
        Hqp[I * Neff:(I + 1) * Neff, I * Neff:(I + 1) * Neff] = -p["lam"][I]
    Hqp[:Neff, Neff:] = -t * np.outer(p["R"][0], p["R"][1])
    Hqp[Neff:, :Neff] = -t * np.outer(p["R"][1], p["R"][0])

    ana, num, agree = {}, {}, {}

    def fd(mod, h):
        pp, pm = mod(+h), mod(-h)
        return (Lfun(pp, ops, t, shift) - Lfun(pm, ops, t, shift)) / (2.0 * h)

    def cp(**kw):
        q = {k: ([x.copy() for x in v] if isinstance(v, list) else v.copy()) for k, v in p.items()}
        for k, v in kw.items():
            q[k] = v
        return q

    # ---- analytic gradients -------------------------------------------------
    imp = [imp_terms(p["phi"][I], ops) for I in range(2)]
    S = [S_of(p["Delta"][I], shift) for I in range(2)]
    gV = [4.0 * (imp[I][1] - S[I] @ p["R"][I]) for I in range(2)]
    glc = [2.0 * (p["Delta"][I] + imp[I][2] - imp[I][3] * np.eye(Neff)) for I in range(2)]
    gl = [2.0 * (p["Delta"][I] - rho[I * Neff:(I + 1) * Neff, I * Neff:(I + 1) * Neff]) for I in range(2)]
    gR = [-4.0 * t * rho[:Neff, Neff:] @ p["R"][1] - 4.0 * S[0] @ p["V"][0],
          -4.0 * t * rho[Neff:, :Neff] @ p["R"][0] - 4.0 * S[1] @ p["V"][1]]
    gD = [2.0 * (p["lam"][I] + p["lamc"][I]) - 2.0 * g.grad_S(p["Delta"][I], np.outer(p["V"][I], p["R"][I]), shift)
          for I in range(2)]
    comm = Hqp @ rho - rho @ Hqp
    ana["R"] = max(abs(x).max() for x in gR)
    ana["V"] = max(abs(x).max() for x in gV)
    # symmetric blocks: report the directional derivative along E_ab + E_ba (= g_ab + g_ba),
    # the same quantity the finite differences below measure
    dirmax = lambda gs: max(abs(float(np.sum(x * d))) for x in gs for d in sym_dirs(Neff))
    ana["Delta"] = dirmax(gD)
    ana["lam"] = dirmax(gl)
    ana["lamc"] = dirmax(glc)
    ana["Psi ([Hqp,rho])"] = abs(comm).max()
    gphi = []
    for I in range(2):
        Hs = imp_sector_H(p["V"][I], p["lamc"][I], ops)
        phi = p["phi"][I]
        gphi.append(2.0 * (Hs @ phi - (phi @ Hs @ phi) * phi))
    ana["Phi (H-<H>)phi"] = max(abs(x).max() for x in gphi)

    # ---- numerical gradients (central differences of L) -------------------
    def worst(vals):
        return max(abs(v) for v in vals)

    vals = {"R": [], "V": [], "lam": [], "lamc": [], "Delta": [], "Psi": [], "Phi": []}
    h = 1e-6
    for I in range(2):
        for a in range(Neff):
            def modR(s, I=I, a=a):
                R = [x.copy() for x in p["R"]]; R[I][a] += s
                return cp(R=R)
            vals["R"].append(fd(modR, h))
            def modV(s, I=I, a=a):
                V = [x.copy() for x in p["V"]]; V[I][a] += s
                return cp(V=V)
            vals["V"].append(fd(modV, h))
        for d in sym_dirs(Neff):
            for key in ("lam", "lamc"):
                def modM(s, I=I, d=d, key=key):
                    M = [x.copy() for x in p[key]]; M[I] = M[I] + s * d
                    return cp(**{key: M})
                vals[key].append(fd(modM, h))
    # Delta: nonlinear (S has a 1/sqrt(shift) scale near n=0,1) -> scan the step and keep the
    # step whose FD best agrees with the analytic directional derivative
    best_dev = 0.0
    for I in range(2):
        for d in sym_dirs(Neff):
            an = float(np.sum(gD[I] * d))
            cands = []
            for hh in hs:
                def modD(s, I=I, d=d):
                    D = [x.copy() for x in p["Delta"]]; D[I] = D[I] + s * d
                    return cp(Delta=D)
                cands.append(fd(modD, hh))
            devs = [abs(c - an) if np.isfinite(c) else np.inf for c in cands]
            k = int(np.argmin(devs))
            vals["Delta"].append(cands[k] if np.isfinite(cands[k]) else np.nan)
            best_dev = max(best_dev, devs[k])
    # Psi: rotations occupied -> virtual
    e, W = np.linalg.eigh(Hqp)
    for a in range(Neff):
        for i in range(Neff):
            A = np.outer(W[:, Neff + a], W[:, i]) - np.outer(W[:, i], W[:, Neff + a])
            def modC(s, A=A):
                return cp(C=expm(s * A) @ p["C"])
            vals["Psi"].append(fd(modC, h))
    # Phi: tangent gradient on the unit sphere
    for I in range(2):
        phi = p["phi"][I]
        gfd = np.zeros_like(phi)
        for k in range(len(phi)):
            def modP(s, I=I, k=k):
                ph = [x.copy() for x in p["phi"]]; ph[I][k] += s
                return cp(phi=ph)
            gfd[k] = fd(modP, 1e-5)
        gfd = gfd - (phi @ gfd) * phi
        vals["Phi"].extend(gfd.tolist())
        agree[f"Phi{I}: |FD - analytic|"] = abs(gfd - gphi[I]).max()

    num["R"], num["V"], num["Delta"] = worst(vals["R"]), worst(vals["V"]), worst(vals["Delta"])
    num["lam"], num["lamc"] = worst(vals["lam"]), worst(vals["lamc"])
    num["Psi ([Hqp,rho])"], num["Phi (H-<H>)phi"] = worst(vals["Psi"]), worst(vals["Phi"])
    agree["Delta: max|FD - analytic| (best step)"] = best_dev
    return num, ana, agree


def converged_point(R0, R1, L0, L1, U, t=1.0, shift=1e-9, tol_mat=1e-8, round_fcidump=False):
    """Run gga_dimer to tight convergence and assemble the variables of L."""
    Neff = len(R0)
    res = g.run_gga(Neff - 1, U, t=t, Rg0=R0, Rg1=R1, lamg0=L0, lamg1=L1, max_iter=300,
                    tol_E=0.0, tol_mat=tol_mat, sqmat_shift=shift, round_fcidump=round_fcidump)
    ops = g.build_impurity_ops(Neff, U, U / 2.0)
    R = [res["R0"], res["R1"]]
    lam = [res["lam0"], res["lam1"]]
    Hqp, rho = g.qp_state(R, lam, t)
    e, W = np.linalg.eigh(Hqp)
    C = W[:, :Neff]
    Delta = [rho[:Neff, :Neff], rho[Neff:, Neff:]]
    V = [res["V0"], res["V1"]]
    lamc = [res["lamc0"], res["lamc1"]]
    phi = []
    for I in range(2):
        H = imp_sector_H(V[I], lamc[I], ops)
        ev, evec = np.linalg.eigh(H)
        phi.append(evec[:, 0])
    p = dict(R=[x.copy() for x in R], Delta=[x.copy() for x in Delta], V=[x.copy() for x in V],
             lam=[x.copy() for x in lam], lamc=[x.copy() for x in lamc], C=C, phi=phi)
    return p, ops, res


def lamc_split(p, ops, tol=1e-6):
    """Split the lamc-gradient 2(Delta + Dbb - 1) by the eigen-directions of Delta.
    A direction is INERT if n < tol or n > 1-tol and R, V have no component on it
    (a sleeping ghost); otherwise ACTIVE."""
    Neff = ops["Neff"]
    act, ine, info = 0.0, 0.0, []
    for I in range(2):
        n, W = np.linalg.eigh(p["Delta"][I])
        Hs = imp_sector_H(p["V"][I], p["lamc"][I], ops)
        phi = np.linalg.eigh(Hs)[1][:, 0]
        _, _, Dbb, nrm = imp_terms(phi, ops)
        Gt = W.T @ (2.0 * (p["Delta"][I] + Dbb - nrm * np.eye(Neff))) @ W
        Rt, Vt = W.T @ p["R"][I], W.T @ p["V"][I]
        inert = [bool((n[a] < tol or n[a] > 1 - tol) and abs(Rt[a]) < tol and abs(Vt[a]) < tol)
                 for a in range(Neff)]
        A = [a for a in range(Neff) if not inert[a]]
        B = [a for a in range(Neff) if inert[a]]
        if A:
            act = max(act, np.abs(Gt[np.ix_(A, A)]).max())
        if B:
            ine = max(ine, np.abs(Gt[B, :]).max())
        info.append((n.round(6), inert))
    return act, ine, info


def report(title, p, ops, t, shift):
    num, ana, agree = analysis(p, ops, t, shift)
    print(f"  {title}")
    print(f"    {'block':22s} {'max|dL/dx| numeric':>20s} {'max|dL/dx| analytic':>22s}")
    for k in num:
        print(f"    {k:22s} {num[k]:20.3e} {ana[k]:22.3e}")
    for k, v in agree.items():
        print(f"    {k:44s} {v:.2e}")
    act, ine, info = lamc_split(p, ops)
    print(f"    dL/dlamc on ACTIVE directions: {act:.2e};  on directions touching INERT ghosts: {ine:.2e}")
    for I, (n, inert) in enumerate(info):
        print(f"      fragment {I}: n = {n.tolist()}, inert = {inert}")
    return num, ana


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: check_stationarity.py <R.in> <L.in> <U>   (Ng = Neff-1 is read from R.in)")
        sys.exit(1)
    R0, R1 = g.read_R_guess(sys.argv[1])
    L0, L1 = g.read_lambda_guess(sys.argv[2])
    U = float(sys.argv[3])
    t, shift = 1.0, 1e-9
    p, ops, res = converged_point(R0, R1, L0, L1, U, t, shift)
    print(f"converged: {res['converged']} (iters={res['iters'] + 1}),  E_var = {res['E_var']:.10f},  "
          f"Z0 = {res['Z0']:.6f}, Z1 = {res['Z1']:.6f}")
    print(f"  L at the converged point = {Lfun(p, ops, t, shift):.10f}   (should equal E_var)")
    report("STATIONARITY at the converged solution", p, ops, t, shift)
    q = {k: ([x.copy() for x in v] if isinstance(v, list) else v.copy()) for k, v in p.items()}
    q["R"][0][0] += 1e-2
    report("SENSITIVITY: same point with R^0[0] shifted by +1e-2 (must be non-stationary)", q, ops, t, shift)
