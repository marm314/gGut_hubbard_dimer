"""U scan with FOUR ghosts (Ng=4, Neff=5) by the direct minimization of not_imp_gga_dimer.py,
U/t = 0.1, 0.2, ..., 10.0.

Random starts at Ng=4 only reach the higher inert-ghost boundary minima, so two starts are
used at every U and the lower converged result is kept:

  embedded   the lowest-energy Ng=2 solution of scan_not_imp/ at this U, embedded in Ng=4 by
             adding two inert bath orbitals per fragment (impurity: one filled, one empty;
             qp: the opposite, so Delta = 1 - Dbb still holds), then perturbed by 1e-3
             (at the exact embedded point the constraint Jacobian is rank deficient and
             SLSQP stops immediately);
  continued  the converged Ng=4 variables of the previous U (constraints do not depend on U).

Each point is minimized until |c| < 1e-9 and the projected gradient < 5e-6 (up to 5 SLSQP
passes), or the pass limit is reached (then flagged).

Output in scan_ng4/: U_<U>/ (output.out, eqp.txt (10 levels), not_imp_Ng4_U*_{R,lambda,lambda_c}.dat,
z.npy), summary.txt, E_vs_U.png (with the Ng=2 energies for comparison and the difference),
eqp_vs_U.png.
"""
import contextlib
import io
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gga_dimer as g
import not_imp_gga_dimer as ni

Ng, t = 4, 1.0
OUT = "scan_ng4"
SC2 = "scan_not_imp"


def embed_ng2_in_ng4(m2, m4, z2):
    N2, N4 = m2.N, m4.N
    sec2, sec4 = m2.ops["sector"], m4.ops["sector"]
    idx4 = sec2 * 16 + 12       # appended modes d3(up,dn) occupied, d4(up,dn) empty (JW order: last 4 bits)
    pos = np.searchsorted(sec4, idx4)
    assert np.all(sec4[pos] == idx4)
    x0, x1, X2 = m2.unpack(z2)
    xs = []
    for x in (x0, x1):
        v = np.zeros(m4.m)
        v[pos] = x
        xs.append(v)
    X4 = np.zeros((2 * N4, N4))
    for f in range(2):
        for o in range(N2):
            X4[f * N4 + o, :N2] = X2[f * N2 + o]
    X4[0 * N4 + 4, N2] = 1.0     # qp orbital 4 occupied (impurity bath orbital 4 empty)
    X4[1 * N4 + 4, N2 + 1] = 1.0
    return m4.pack(xs[0], xs[1], X4)


def tangent_grad(model, z):
    E, gE = model.energy(z)
    J = model.constraint_jac(z)
    u, s, vt = np.linalg.svd(J)
    rank = int((s > 1e-9 * s[0]).sum())
    return np.linalg.norm(vt[rank:] @ gE)


def converge(model, z, max_pass=5):
    for k in range(1, max_pass + 1):
        z = model.minimize(z, max_iter=1000).x
        viol = np.abs(model.constraints(z)).max()
        tg = tangent_grad(model, z)
        if viol < 1e-9 and tg < 5e-6:
            break
    return z, k, viol, tg


def lowest_ng2(U):
    for line in open(os.path.join(SC2, "lowest_energy.txt")):
        if not line.startswith("#") and line.strip():
            p = line.split()
            if abs(float(p[0]) - U) < 1e-6:
                return p[2], float(p[1])


def solve_point(U, cands):
    """Minimize from every candidate start; keep the lowest feasible result."""
    m4 = ni.NotImpDimer(Ng, U, t)
    best = None
    for name, zs in cands.items():
        z, npass, viol, tg = converge(m4, zs)
        E = m4.energy(z, grad=False)
        if viol < 1e-8 and (best is None or E < best[0] - 1e-12):
            best = (E, z, name, npass, viol, tg)
    if best is None:
        raise RuntimeError(f"no feasible Ng=4 solution at U={U}")
    return m4, best


def save_point(U, m4, best, E2):
    E4, z, name, npass, viol, tg = best
    a = m4.analyse(z)
    d = os.path.join(OUT, f"U_{U:.1f}")
    os.makedirs(d, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        g.save_results(a, prefix=os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}"))
    np.savetxt(os.path.join(d, "eqp.txt"), a["eqp"], header=f"quasiparticle energies, Ng=4, U/t={U:.4f}")
    np.save(os.path.join(d, "z.npy"), z)
    conv = viol < 1e-9 and tg < 5e-6
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(f"Ng={Ng}  U/t={U:.4f}  Z0={a['Z0']:.6f}  Z1={a['Z1']:.6f}  E_var/t={E4:.8f}  docc={a['docc']:.6f}  "
                f"start={name}  passes={npass}  |c|={viol:.2e}  |tangent grad|={tg:.2e}  converged={conv}  "
                f"E(Ng=2)={E2:.8f}  E4-E2={E4-E2:.2e}\n")
    return (U, E4, E2, a["Z0"], a["Z1"], a["docc"], viol, tg, a["eqp"], name, conv)


def load_rows():
    import re
    rows = []
    for U in [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]:
        d = os.path.join(OUT, f"U_{U:.1f}")
        line = open(os.path.join(d, "output.out")).read()
        gv = lambda key: float(re.search(key + r"=(-?[\d.]+(?:e[+-]?\d+)?)", line).group(1))
        rows.append((U, gv("E_var/t"), gv(r"E\(Ng=2\)"), gv("Z0"), gv("Z1"), gv("docc"), gv(r"\|c\|"),
                     gv(r"\|tangent grad\|"), np.loadtxt(os.path.join(d, "eqp.txt")),
                     re.search(r"start=(\w+)", line).group(1), "converged=True" in line))
    return rows


def finalize(rows, t0):
    U = np.array([r[0] for r in rows])
    E4 = np.array([r[1] for r in rows]); E2 = np.array([r[2] for r in rows])
    np.savetxt(os.path.join(OUT, "summary.txt"),
               np.array([[r[0], r[1], r[2], r[1] - r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows]),
               header="U/t E_var(Ng=4) E_var(Ng=2) E4-E2 Z0 Z1 docc max|c| |tangent grad|")
    eq = np.array([r[8] for r in rows])
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.column_stack([U, eq]), header="U/t eqp0..eqp9 (Ng=4)")

    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, E4, "-", lw=1.6, label="Ng=4 (direct minimization)")
    ax[0].plot(U, E2, "--", lw=1.2, label="Ng=2 (direct minimization, lowest)")
    ax[0].set_ylabel("E_var / t"); ax[0].set_title("Variational energy vs U"); ax[0].legend(fontsize=8)
    ax[1].plot(U, E4 - E2, "-", lw=1.2, color="tab:red")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("E(Ng=4) - E(Ng=2)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "E_vs_U.png"), dpi=150); plt.close()

    plt.figure(figsize=(6.4, 4.8))
    for i in range(eq.shape[1]):
        plt.plot(U, eq[:, i], "-", lw=1.1)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title("Quasiparticle energies vs U (Ng=4, direct minimization)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150); plt.close()

    nb = sum(1 for r in rows if not r[10])
    print(f"\nmax |E4-E2| = {np.abs(E4 - E2).max():.3e};  min(E4-E2) = {(E4 - E2).min():.3e};  "
          f"points not fully converged: {nb};  total {time.time()-t0:.0f}s")


def start_candidates(U, m2, m4, z2, z_prev, z_next=None):
    rng = np.random.default_rng(int(round(U * 10)))
    zemb = embed_ng2_in_ng4(m2, m4, z2)
    c = {"embedded": zemb + 1e-3 * rng.normal(size=m4.nz)}
    if z_prev is not None:
        c["continued"] = z_prev
    return c


if __name__ == "__main__":
    import sys
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]
    refine = "--refine" in sys.argv

    if not refine:
        rows, z_prev = [], None
        for U in Us:
            m2 = ni.NotImpDimer(2, U, t)
            branch, E2 = lowest_ng2(U)
            z2 = np.load(os.path.join(SC2, branch, f"U_{U:.1f}", "z.npy"))
            m4, best = solve_point(U, start_candidates(U, m2, ni.NotImpDimer(Ng, U, t), z2, z_prev))
            z_prev = best[1]
            r = save_point(U, m4, best, E2)
            rows.append(r)
            print(f"U/t={U:5.1f}  E(Ng=4)={r[1]: .8f}  E(Ng=2)={E2: .8f}  diff={r[1]-E2: .2e}  Z={r[3]:.5f}  "
                  f"start={r[9]:9s} passes={best[3]} |c|={r[6]:.0e} tg={r[7]:.0e}"
                  f"{'' if r[10] else '  <-- not fully converged'}   [{time.time()-t0:.0f}s]", flush=True)
        finalize(rows, t0)
    else:
        # re-solve the points where Ng=4 ended ABOVE the (embeddable) Ng=2 solution, with extra starts:
        # smaller embedded perturbations and the converged Ng=4 variables of both neighbouring U
        rows = load_rows()
        for k, r in enumerate(rows):
            U, E4, E2 = r[0], r[1], r[2]
            if E4 - E2 <= 1e-8:
                continue
            m2, m4 = ni.NotImpDimer(2, U, t), ni.NotImpDimer(Ng, U, t)
            branch, _ = lowest_ng2(U)
            z2 = np.load(os.path.join(SC2, branch, f"U_{U:.1f}", "z.npy"))
            zemb = embed_ng2_in_ng4(m2, m4, z2)
            rng = np.random.default_rng(12345)
            cands = {f"embedded_{eps:g}": zemb + eps * rng.normal(size=m4.nz) for eps in (1e-4, 3e-4, 3e-3)}
            for name, dU in (("from_below", -0.1), ("from_above", +0.1)):
                Un = round(U + dU, 1)
                if 0.1 <= Un <= 10.0:
                    cands[name] = np.load(os.path.join(OUT, f"U_{Un:.1f}", "z.npy"))
            m4, best = solve_point(U, cands)
            if best[0] < E4 - 1e-12:
                rows[k] = save_point(U, m4, best, E2)
                print(f"U/t={U:4.1f}: refined  E4-E2 {E4-E2: .2e} -> {best[0]-E2: .2e}  (start {best[2]})", flush=True)
            else:
                print(f"U/t={U:4.1f}: no lower solution found (E4-E2 stays {E4-E2: .2e})", flush=True)
        finalize(rows, t0)
