"""U scan with FOUR ghosts (Ng=4, Neff=5), Delta V/t=2.0, by the direct minimization of
not_imp_gga_dimer.py -- the Delta V=2 analogue of scan_ng4_U.py, kept as a SEPARATE
script/output folder so the existing Delta V=0 results are untouched. Identical algorithm
to scan_ng4_U.py (see its docstring); only deltaV and the input/output folders differ.

Random starts at Ng=4 only reach the higher inert-ghost boundary minima, so two starts are
used at every U and the lower converged result is kept:

  embedded   the lowest-energy Ng=2 solution of scan_ng2_not_imp_dV2/ at this U, embedded in
             Ng=4 by adding two inert bath orbitals per fragment (impurity: one filled, one
             empty; qp: the opposite, so Delta = 1 - Dbb still holds), then perturbed by
             1e-3 (at the exact embedded point the constraint Jacobian is rank deficient and
             SLSQP stops immediately);
  continued  the converged Ng=4 variables of the previous U (constraints do not depend on U).

Each point is minimized until |c| < 1e-9 and the projected gradient < 5e-6 (up to 5 SLSQP
passes), or the pass limit is reached (then flagged).

Output in scan_ng4_not_imp_dV2/: U_<U>/ (output.out, eqp.txt (10 levels),
not_imp_Ng4_U*_{R,lambda,lambda_c}.dat, z.npy), summary.txt, E_vs_U.png (with the Ng=2
energies for comparison and the difference), eqp_vs_U.png.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gga_dimer as g
import not_imp_gga_dimer as ni

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scan_ed_deltaV"))
import make_plot as ed_dV  # scan_ed_deltaV/make_plot.py

# eqp_vs_U.png colors: one color per +-symmetric pair (outermost first), not per raw sorted
# eigenvalue index -- see scan_ng2_not_imp_dV2_U.py's branch_color for the full rationale.
PAIR_COLORS = ["tab:orange", "tab:blue"]
GHOST_COLOR = "0.6"


def branch_color(i, n):
    pair = min(i, n - 1 - i)
    return PAIR_COLORS[pair] if pair < len(PAIR_COLORS) else GHOST_COLOR

Ng, t, deltaV = 4, 1.0, 2.0
OUT = "scan_ng4_not_imp_dV2"
SC2 = "scan_ng2_not_imp_dV2"


def embed_ng2_in_ng4(m2, m4, z2):
    N2, N4 = m2.N, m4.N
    sec2, sec4 = m2.ops[0]["sector"], m4.ops[0]["sector"]
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
    m4 = ni.NotImpDimer(Ng, U, t, deltaV=deltaV)
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
    np.savetxt(os.path.join(d, "eqp.txt"), a["eqp"],
               header=f"quasiparticle energies, Ng=4, U/t={U:.4f}, deltaV/t={deltaV:.4f}")
    np.save(os.path.join(d, "z.npy"), z)
    conv = viol < 1e-9 and tg < 5e-6
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(f"Ng={Ng}  U/t={U:.4f}  deltaV/t={deltaV:.4f}  Z0={a['Z0']:.6f}  Z1={a['Z1']:.6f}  "
                f"E_var/t={E4:.8f}  docc={a['docc']:.6f}  start={name}  passes={npass}  "
                f"|c|={viol:.2e}  |tangent grad|={tg:.2e}  converged={conv}  "
                f"E(Ng=2)={E2:.8f}  E4-E2={E4-E2:.2e}\n")
    return (U, E4, E2, a["Z0"], a["Z1"], a["docc"], viol, tg, a["eqp"], name, conv)


def finalize(rows, t0):
    U = np.array([r[0] for r in rows])
    E4 = np.array([r[1] for r in rows]); E2 = np.array([r[2] for r in rows])
    np.savetxt(os.path.join(OUT, "summary.txt"),
               np.array([[r[0], r[1], r[2], r[1] - r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows]),
               header="U/t E_var(Ng=4) E_var(Ng=2) E4-E2 Z0 Z1 docc max|c| |tangent grad|")
    eq = np.array([r[8] for r in rows])
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.column_stack([U, eq]), header="U/t eqp0..eqp9 (Ng=4)")

    # deltaV_ed = 2*deltaV, NOT deltaV: exact_dimer_deltaV.py's deltaV and gga_dimer.py's
    # deltaV are different physical parameters (see scan_ng2_not_imp_dV2_U.py's
    # ed_ground_state_energy docstring) -- getting this wrong makes the gGA energy appear
    # to violate the variational principle (caught exactly that way on the Ng=2 scan).
    E_ed = np.array([ed_dV.E0_explicit(t, u, 2.0 * deltaV) for u in U])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, E4, "-", lw=1.6, label="Ng=4")
    ax[0].plot(U, E2, "--", lw=1.2, label="Ng=2 (lowest)")
    ax[0].plot(U, E_ed, ":", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("E_var / t")
    ax[0].set_title(f"Variational energy vs U ($\\Delta V/t={deltaV:g}$)"); ax[0].legend(fontsize=8)
    ax[1].plot(U, E4 - E2, "-", lw=1.2, color="tab:red", label="Ng4-Ng2")
    ax[1].plot(U, E4 - E_ed, "-", lw=1.2, color="tab:purple", label="Ng4-ED")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("energy difference"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "E_vs_U.png"), dpi=150); plt.close()

    # deltaV_ed = 2*deltaV, NOT deltaV -- see scan_ng2_not_imp_dV2_U.py's
    # ed_ground_state_energy docstring.
    ed_p = np.array([sorted(ed_dV.analytic_poles(t, u, 2.0 * deltaV)[0]) for u in U])
    n = eq.shape[1]
    plt.figure(figsize=(6.4, 4.8))
    for i in range(n):
        plt.plot(U, eq[:, i], "-", lw=1.1, color=branch_color(i, n))
    for i in range(ed_p.shape[1]):
        plt.plot(U, ed_p[:, i], "--", lw=1.2, color="black",
                  label="exact (ED)" if i == 0 else None)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title(f"Quasiparticle energies vs U (Ng=4, $\\Delta V/t={deltaV:g}$)")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150); plt.close()

    nb = sum(1 for r in rows if not r[10])
    print(f"\nmax |E4-E2| = {np.abs(E4 - E2).max():.3e};  min(E4-E2) = {(E4 - E2).min():.3e};  "
          f"points not fully converged: {nb};  total {time.time()-t0:.0f}s")


def start_candidates(U, m2, m4, z2, z_prev):
    rng = np.random.default_rng(int(round(U * 10)))
    zemb = embed_ng2_in_ng4(m2, m4, z2)
    c = {"embedded": zemb + 1e-3 * rng.normal(size=m4.nz)}
    if z_prev is not None:
        c["continued"] = z_prev
    return c


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]

    rows, z_prev = [], None
    for U in Us:
        m2 = ni.NotImpDimer(2, U, t, deltaV=deltaV)
        branch, E2 = lowest_ng2(U)
        z2 = np.load(os.path.join(SC2, branch, f"U_{U:.1f}", "z.npy"))
        m4, best = solve_point(U, start_candidates(U, m2, ni.NotImpDimer(Ng, U, t, deltaV=deltaV), z2, z_prev))
        z_prev = best[1]
        r = save_point(U, m4, best, E2)
        rows.append(r)
        print(f"U/t={U:5.1f}  E(Ng=4)={r[1]: .8f}  E(Ng=2)={E2: .8f}  diff={r[1]-E2: .2e}  Z={r[3]:.5f}  "
              f"start={r[9]:9s} passes={best[3]} |c|={r[6]:.0e} tg={r[7]:.0e}"
              f"{'' if r[10] else '  <-- not fully converged'}   [{time.time()-t0:.0f}s]", flush=True)
    finalize(rows, t0)
