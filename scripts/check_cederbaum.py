"""SI Sec. S5 checks for the Cederbaum gGut implementation (gga_cederbaum.py), c-v hopping.

 A. derivative identities at RANDOM points (any B_c, no convergence needed): the formulas
    for H_qp, S V, lambda+lambda^c (G, Eq. lamc3), alpha and Xi against central finite
    differences of the energy functional E, Eq. (Efunc).
 B. at a converged solution: finite-difference gradients of the Lagrangian L with respect to
    Phi_I (built independently from the definitions, raw Fock operators), Delta_I; the qp
    stationarity [H_qp, rho]; the constraints residual.
 C. limits: V=0 (exact non-interacting energy) and t_c=0 (E=-2Ec-2Delta, nu_c=2).
"""
import sys
import numpy as np
from gga_dimer import grad_S, _sqrt_eig, apply_S_inv
import gga_cederbaum as g
import exact_cederbaum as ed

H_FD = 1e-6
rng = np.random.default_rng(7)


def rsym(n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * 0.5 * (a + a.T)


def rdens(n, lo=0.15, hi=0.85):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * rng.uniform(lo, hi, n)) @ q.T


def fd(f, x, h=H_FD):
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------------------
def check_A(Bc, hop, shift=1e-9):
    m = g.Model(1.0, 10.0, 1.3, 0.4, Bc, shift, hop)
    n = m.n
    rho = rsym(n, 0.3)
    R = [rng.uniform(0.3, 1, Bc), rng.uniform(0.3, 1, 1), rng.uniform(0.3, 1, 1)]
    D = [rdens(Bc), rdens(1), rdens(1)]
    nu = 1.7
    K = rsym(Bc, 0.5) + 2 * nu / Bc * np.eye(Bc)
    Eloc = -3.0
    lam = [rsym(Bc), rsym(1), rsym(1)]
    E = lambda rho_, R_, Dc_, nu_, K_: m.energy(rho_, R_, Dc_, nu_, K_, Eloc)
    out = {}

    u, w = m.uw(rho, R)
    Tb = m.theta_bar(D[0], nu, K)
    H = m.build_Hqp(R, lam, Tb, u, w, nu)
    # 1. H_qp off-diagonal blocks = 1/4 dE/d rho^{IJ}
    dev = 0.0
    for (I, J) in ((0, 1), (0, 2), (1, 2)):
        for a in range(m.B[I]):
            for b in range(m.B[J]):
                def f(x, I=I, J=J, a=a, b=b):
                    r = rho.copy(); r[m.sl[I].start + a, m.sl[J].start + b] += x
                    r[m.sl[J].start + b, m.sl[I].start + a] += x
                    return E(r, R, D[0], nu, K)
                dev = max(dev, abs(0.25 * fd(f, 0.0) - H[m.sl[I], m.sl[J]][a, b]))
    out["H_qp blocks vs 1/4 dE/drho"] = dev

    # 2. S_I V_I = 1/4 dE/dR_I
    hyb = m.fit_hyb(rho, R, D, Tb, u, w, nu)
    dev = 0.0
    for I in range(3):
        S = (lambda Dm: (lambda d, W, s: (W * s) @ W.T)(*_sqrt_eig(Dm, shift)))(D[I])
        for a in range(m.B[I]):
            def f(x, I=I, a=a):
                R2 = [r.copy() for r in R]; R2[I][a] += x
                return E(rho, R2, D[0], nu, K)
            dev = max(dev, abs(0.25 * fd(f, 0.0) - (S @ hyb[I])[a]))
    out["S V vs 1/4 dE/dR"] = dev

    # 3. G = 2 grad^s Tr[M S(Delta)]   and   -1/2 grad^s E = lam_sum - G(Delta, V R^T)
    ls = m.lam_sum(D, hyb, R, u, w, nu, K)
    devG, devE = 0.0, 0.0
    for I in range(3):
        M = rng.standard_normal((m.B[I], m.B[I]))
        Bd = rsym(m.B[I])
        Sfun = lambda Dm: (lambda d, W, s: (W * s) @ W.T)(*_sqrt_eig(Dm, shift))
        fdv = fd(lambda x: np.trace(M @ Sfun(D[I] + x * Bd)), 0.0)
        devG = max(devG, abs(0.5 * np.trace(grad_S(D[I], M, shift) @ Bd) - fdv))
    out["G vs FD of Tr[M S]"] = devG
    Bd = rsym(Bc)
    fdE = fd(lambda x: E(rho, R, D[0] + x * Bd, nu, K), 0.0)
    gradE = 2.0 * (grad_S(D[0], np.outer(hyb[0], R[0]), shift) - ls[0])   # E-gradient implied by lamc3
    out["lamc3 (core) vs FD dE/dDelta"] = abs(np.trace(gradE @ Bd) - fdE)
    # o, v: E does not depend on their Delta -> lam_sum == G exactly
    out["lam_sum(o,v) - G"] = max(np.abs(ls[I] - grad_S(D[I], np.outer(hyb[I], R[I]), shift)).max() for I in (1, 2))

    # 4. alpha = dE/dnu, Xi = dE/dK
    alpha, Xi = m.core_extra(rho, R, D[0])
    out["alpha vs dE/dnu"] = abs(fd(lambda x: E(rho, R, D[0], nu + x, K), 0.0) - alpha)
    Bk = rsym(Bc)
    out["Xi vs dE/dK"] = abs(fd(lambda x: E(rho, R, D[0], nu, K + x * Bk), 0.0) - np.sum(Xi * Bk))
    return out


# ---------------------------------------------------------------------------
def raw_phi_quantities(fr, phi_sec):
    """nu, K, <Hloc>, D, Dbb, <sum lam dd> from the FULL Fock-space sparse operators."""
    ops = fr.ops; B = fr.B
    full = np.zeros(ops["H_loc"].shape[0]); full[fr.sec] = phi_sec
    ex = lambda A: full @ (A @ full)
    nph = ops["n_pu"] + ops["n_pd"]
    nu = ex(nph)
    cu, cd = ops["c_pu"], ops["c_pd"]
    bu, bd = ops["bath_up"], ops["bath_dn"]
    K = np.array([[ex(nph @ (bu[d] @ bu[g].conj().T + bd[d] @ bd[g].conj().T)) for d in range(B)] for g in range(B)])
    D = np.array([0.5 * (ex(bu[a].conj().T @ cu) + ex(bd[a].conj().T @ cd)) for a in range(B)])
    Dbb = np.array([[0.5 * (ex(bu[a].conj().T @ bu[b]) + ex(bd[a].conj().T @ bd[b])) for b in range(B)] for a in range(B)])
    return nu, K, ex(ops["H_loc"]), D, Dbb, full


def check_B(sol):
    m = g.Model(sol["Delta"], sol["Ec"], sol["Vint"], sol["tc"], sol["Bc"], sol["shift"], sol["hop"])
    R, Dl, lam, lamc, hyb, rho = sol["R"], sol["Dl"], sol["lam"], sol["lamc"], sol["hyb"], sol["rho"]
    out = {}
    # constraints and qp stationarity
    out["||rho^II - (1 - Dbb)||"] = sol["resid"]
    out["||Delta - rho^II||"] = sum(np.linalg.norm(Dl[I] - rho[m.sl[I], m.sl[I]]) for I in range(3))
    out["||R - S^-1 D||"] = sum(np.linalg.norm(sol["Rnew"][I] - R[I]) for I in range(3))
    out["||[H_qp, rho]||"] = np.linalg.norm(sol["Hqp"] @ rho - rho @ sol["Hqp"])
    e = sol["eps"]; out["qp gap e[Nocc]-e[Nocc-1]"] = e[m.Nocc] - e[m.Nocc - 1]
    # Phi gradients: build L_Phi independently
    imp = sol["imp"]
    nu0, K0, Eloc_c, _, _, _ = raw_phi_quantities(m.frag[0], imp[0]["gs"])
    Eloc_others = sum(raw_phi_quantities(m.frag[I], imp[I]["gs"])[2] for I in (1, 2))
    for I in range(3):
        fr = m.frag[I]
        Ec_mult = imp[I]["E"]                      # e_solver = E^c + 2 Tr lam^c
        def ell(phi_sec, I=I, fr=fr):
            nu, K, Eloc, D, Dbb, _ = raw_phi_quantities(fr, phi_sec)
            val = 4.0 * hyb[I] @ D + 2.0 * np.trace(lamc[I] @ Dbb) - (2.0 * np.trace(lamc[I]) + (Ec_mult - 2.0 * np.trace(lamc[I]))) * (phi_sec @ phi_sec)
            if I == 0:
                val += m.energy(rho, R, Dl[0], nu, K, Eloc + Eloc_others)
            else:
                val += Eloc
            return val
        phi0 = imp[I]["gs"]
        worst = 0.0
        for _ in range(8):
            v = rng.standard_normal(phi0.size)
            worst = max(worst, abs(fd(lambda x: ell(phi0 + x * v), 0.0)))
        out[f"dL/dPhi_{'cov'[I]} (8 random dirs)"] = worst
    # Delta gradients: L_Delta(Delta_I) = E(Delta_c) - 4 V_I^T S(Delta_I) R_I + 2 Tr[(lam+lamc) Delta_I]
    Sfun = lambda Dm: (lambda d, W, s: (W * s) @ W.T)(*_sqrt_eig(Dm, m.shift))
    for I in range(3):
        worst = 0.0
        for _ in range(6):
            Bd = rsym(m.B[I], 1e-2)
            def LD(x, I=I, Bd=Bd):
                Dm = Dl[I] + x * Bd
                val = -4.0 * hyb[I] @ Sfun(Dm) @ R[I] + 2.0 * np.trace((lam[I] + lamc[I]) @ Dm)
                if I == 0:
                    val += m.energy(rho, R, Dm, sol["nu"], sol["K"], sum(i["Eloc"] for i in imp))
                return val
            worst = max(worst, abs(fd(LD, 0.0, 1e-7)))
        out[f"dL/dDelta_{'cov'[I]} (6 random dirs)"] = worst
    return out


def show(title, d, tol=None):
    print(f"\n{title}")
    for k, v in d.items():
        print(f"   {k:<42s} {v:.3e}")


if __name__ == "__main__":
    print("== A. derivative identities at random points (hop='cv') ==")
    for Bc in (1, 3, 5):
        show(f"B_c = {Bc}", check_A(Bc, "cv"))
    print("\n== A'. same with hop='co' (original model), B_c=3 ==")
    show("B_c = 3", check_A(3, "co"))

    print("\n== C. limits (B_c=1) ==")
    for V, tc in ((0.0, 0.5), (0.0, 0.1), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)):
        s = g.run(1.0, 10.0, V, tc, 1, max_iter=2000, mix=0.5, tol=1e-10)
        ref = ed.noninteracting_energy(1.0, 10.0, tc, "cv") if V == 0 else -22.0
        print(f"   V={V} tc={tc}: E_var={s['E_var']:.10f}  reference={ref:.10f}  diff={s['E_var']-ref:.1e}  "
              f"nu_c={s['nu']:.8f}  conv={s['converged']} resid={s['resid']:.1e}")

    print("\n== B. converged nontrivial solution (B_c=1, hop='cv') ==")
    for V, tc in ((1.0, 0.5), (1.0, 0.1), (2.0, 0.5)):
        s = g.run(1.0, 10.0, V, tc, 1, max_iter=2000, mix=0.5, tol=1e-12)
        r = ed.solve(1.0, 10.0, V, tc, ht="cv")
        print(f"\n -- V={V} tc={tc}: converged={s['converged']} iters={s['iters']} E_var={s['E_var']:.8f} (ED {r['E0']:.8f}) nu_c={s['nu']:.6f} (ED {r['nu_c']:.6f})")
        show("B checks", check_B(s))
