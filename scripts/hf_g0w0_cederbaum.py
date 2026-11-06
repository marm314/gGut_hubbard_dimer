"""Proper HF + RPA + G0W0@HF for the (t_c=0) Cederbaum model, from the two-body Hamiltonian, compared with the exact hole spectrum and
with the G0W0 of Kocklauner & Golze (satellite at -1.414 Omega, -2.236 Omega for rho/Omega = 0.5, 1.0; measured from their Fig. 2: 1.42, 2.23).

Hamiltonian as one- plus two-body operator (spatial orbitals c,o,v; V enters through -V(2-n_c)T with T = sum_s (v^dag_s o_s + h.c.)):
   h = diag(-E_c, -g, +g) + (-2V) (o-v),     (cc|ov) = (cc|vo) = (ov|cc) = (vo|cc) = V,   all other (pq|rs) = 0,
   H = sum h_pq p^dag q + 1/2 sum (pq|rs) p^dag_s r^dag_s' s_s' q_s .
Step 0 rebuilds H from these integrals and compares with exact_cederbaum.build_hamiltonian (must agree to machine precision).
Step 1 restricted HF (SCF from the bare h). Step 2 direct RPA (A = d_eps + (ia|jb), B = (ia|jb)) in the spin-orbital basis. Step 3 G0W0@HF:
   Sigma_c(w) = sum_{m occ,s} w_s(cm)^2/(w - e_m + Om_s) + sum_{m virt,s} w_s(cm)^2/(w - e_m - Om_s),  w_s(pq) = sum_ia (pq|ia)(X+Y)^s_ia,
the poles/weights of 1/(w - e_c - Sigma_c(w)) from the eigenproblem of the equivalent arrow matrix (hole side: poles < 0).
python3 hf_g0w0_cederbaum.py"""
import numpy as np
import exact_cederbaum as ed

C, O, Vv = 0, 1, 2
def integrals(g, Ec, V):
    h = np.diag([-Ec, -g, g]).astype(float); h[O, Vv] = h[Vv, O] = -2 * V
    I = np.zeros((3, 3, 3, 3))
    for (p, q, r, s) in ((C, C, O, Vv), (C, C, Vv, O), (O, Vv, C, C), (Vv, O, C, C)): I[p, q, r, s] = V
    return h, I

def rebuild(h, I):
    """H from h and (pq|rs) in the Fock space of exact_cederbaum (modes c,o,v up; c,o,v down)."""
    _, c, _ = ed.build_hamiltonian(1.0, 10.0, 0.5, 0.0, "cv")
    cd = [x.T for x in c]; mode = lambda p, s: p + 3 * s
    H = np.zeros((64, 64))
    for s in range(2):
        for p in range(3):
            for q in range(3):
                H += h[p, q] * cd[mode(p, s)] @ c[mode(q, s)]
    for s in range(2):
        for t in range(2):
            for p in range(3):
                for q in range(3):
                    for r in range(3):
                        for u in range(3):
                            if I[p, q, r, u] != 0:
                                H += 0.5 * I[p, q, r, u] * cd[mode(p, s)] @ cd[mode(r, t)] @ c[mode(u, t)] @ c[mode(q, s)]
    return H

def rhf(h, I, nocc=2, iters=500, mix=0.5):
    n = 3; P = np.zeros((n, n)); eps, Cm = np.linalg.eigh(h); Cm = Cm; P = Cm[:, :nocc] @ Cm[:, :nocc].T
    for it in range(iters):
        F = h + 2 * np.einsum("pqrs,rs->pq", I, P) - np.einsum("psrq,rs->pq", I, P)
        eps, Cm = np.linalg.eigh(F); Pn = Cm[:, :nocc] @ Cm[:, :nocc].T
        if np.abs(Pn - P).max() < 1e-13: P = Pn; break
        P = mix * Pn + (1 - mix) * P
    F = h + 2 * np.einsum("pqrs,rs->pq", I, P) - np.einsum("psrq,rs->pq", I, P)
    E = np.sum((h + F) * P) * 1.0
    return eps, Cm, F, E, it

def g0w0(g, Ec, V):
    h, I = integrals(g, Ec, V)
    err = np.abs(rebuild(h, I) - ed.build_hamiltonian(g, Ec, V, 0.0, "cv")[0]).max()
    eps, Cm, F, EHF, it = rhf(h, I)
    # MO integrals (spatial)
    Imo = np.einsum("pqrs,pa,qb,rc,sd->abcd", I, Cm, Cm, Cm, Cm)
    # spin-orbitals: index = (spatial MO, spin); occupied: MOs 0,1 (both spins); virtual: MO 2
    occ = [(i, s) for i in range(2) for s in range(2)]; vir = [(2, s) for s in range(2)]
    ex = [(i, a) for i in occ for a in vir if i[1] == a[1]]                       # spin-conserving singles
    ei = lambda x: eps[x[0]]
    def ints(p, q, r, s_):                                                         # (pq|rs) spin-orbital
        return Imo[p[0], q[0], r[0], s_[0]] if (p[1] == q[1] and r[1] == s_[1]) else 0.0
    n = len(ex); A = np.zeros((n, n)); B = np.zeros((n, n))
    for x, (i, a) in enumerate(ex):
        for y, (j, b) in enumerate(ex):
            A[x, y] = (ei(a) - ei(i)) * (x == y) + ints(i, a, j, b); B[x, y] = ints(i, a, b, j)
    # (A-B)^1/2 (A+B) (A-B)^1/2 Z = Om^2 Z
    wA, UA = np.linalg.eigh(A - B); sq = (UA * np.sqrt(wA)) @ UA.T
    Om2, Z = np.linalg.eigh(sq @ (A + B) @ sq); Om = np.sqrt(Om2)
    XpY = (np.linalg.inv(sq) @ Z) * np.sqrt(Om)[None, :]                           # (X+Y)_s, normalized (X+Y)^T (A-B)^-1... see check
    # normalization check: (X+Y)^T (A-B) (X+Y) = Om
    nrm = np.diag(XpY.T @ (A - B) @ XpY) / Om
    W = lambda p, q: np.array([sum(ints(p, q, i, a) * XpY[x, s] for x, (i, a) in enumerate(ex)) for s in range(n)])
    pc = (0, 0)                                                                    # core, spin up
    e_list, w_list = [], []
    for m in occ + vir:
        w = W(pc, m)
        for s in range(n):
            if abs(w[s]) > 1e-12:
                e_list.append(eps[m[0]] - Om[s] if m in occ else eps[m[0]] + Om[s]); w_list.append(w[s])
    M = np.zeros((len(e_list) + 1, len(e_list) + 1)); M[0, 0] = eps[0]
    for k, (e, w) in enumerate(zip(e_list, w_list)): M[k + 1, k + 1] = e; M[0, k + 1] = M[k + 1, 0] = w
    ev, U = np.linalg.eigh(M); wt = U[0] ** 2; hole = ev < 0
    return dict(err_H=err, eps_HF=eps, E_HF=EHF, scf_it=it, F_ov=F[O, Vv], Om=Om, XpYnorm=nrm, poles=ev[hole], w=wt[hole],
                rho=np.sqrt(sum(w * w for w in w_list)), Omega_bright=Om[np.argmax([abs(W(pc, pc)[s]) for s in range(n)])])

if __name__ == "__main__":
    g, Ec = 1.0, 10.0; out = {}
    for V in (0.5 ** 0.5, 2 ** 0.5):
        r = g0w0(g, Ec, V); ex = ed.solve(g, Ec, V, 0.0, ht="cv"); Omega = 2 * g; rho = np.sqrt(2) * V
        sel = r["w"] > 1e-6; pg = r["poles"][sel]; wg = r["w"][sel]; i0 = np.argmax(wg); qp = pg[i0]
        se = ex["weights"] > 1e-6; pe = ex["poles"][se]; we = ex["weights"][se]; j0 = np.argmax(we)
        an = Omega * np.sqrt(1 + 4 * rho ** 2 / Omega ** 2)
        print(f"V/g={V:.4f} rho/Omega={rho/Omega:.3f}: |H(integrals) - H(exact_cederbaum)| = {r['err_H']:.1e};  RHF: {r['scf_it']} it, eps={np.round(r['eps_HF'],6)}, F_ov={r['F_ov']:.1e}, E_HF={r['E_HF']:.6f}")
        print(f"   RPA excitations Om_s = {np.round(r['Om'],6)} (Omega={Omega}); (X+Y) norm {np.round(r['XpYnorm'],6)}; coupling rho = {r['rho']:.6f} (sqrt2 V = {rho:.6f})")
        print(f"   G0W0@HF hole poles {np.round(pg,5)} w {np.round(wg,5)} -> QP {qp:.5f}, satellite {np.round((pg-qp)/Omega,5)} Omega; analytic -{an/Omega:.5f} Omega; digitized Golze {(-1.41847826, -2.23369565)[int(rho/Omega>0.75)]:.3f}")
        print(f"   exact poles {np.round(pe,4)} w {np.round(we,4)} -> satellites {np.round((pe-pe[j0])/Omega,4)} Omega\n", flush=True)
        out[f"V{V:.4f}"] = np.array([pg, wg, pe[:len(pg)] if False else np.resize(pe, len(pg))], dtype=object)
