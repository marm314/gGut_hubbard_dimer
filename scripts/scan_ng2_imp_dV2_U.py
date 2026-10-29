"""U-continuation scans (Ng=2, Delta V/t=2.0) with gga_dimer.py's iterative embedding --
the Delta V=2 analogue of scan_U_forward.py/scan_U_backward.py combined into one script
(no pre-existing Delta V=2 guess file exists to prime from, so each branch starts cold,
with gga_dimer.run_gga's own default initial R, at its own first U).

Two branches, both by continuation in U (every U uses the previous U's converged (R,
lambda) as the next guess):

  forward   U/t = 0.1 -> 10.0 in steps of +0.1, cold start at U=0.1.
  backward  U/t = 10.0 -> 0.1 in steps of -0.1, cold start at U=10.0.

Used as an independent cross-check of scan_ng2_not_imp_dV2/ (the direct-minimization
scan): the two solvers are built from unrelated numerical methods (self-consistent
embedding cycle vs. constrained direct minimization) and should agree at every U if both
found the true minimum -- a comparison that caught a real discrepancy (scan_ng2_not_imp_dV2's
single-branch version was stuck in a higher-energy fixed point at U/t=5, 8).

For every U it writes scan_ng2_imp_dV2/<branch>/U_<U>/ : R.in, L.in (the guess fed in),
output.out (run_gga trace + summary), gga_Ng2_U*_{R,lambda,lambda_c}.dat, eqp.txt. Per
branch: eqp_vs_U.txt, eqp_vs_U.png. Combined, directly in scan_ng2_imp_dV2/:
lowest_energy.txt (same format as scan_ng2_not_imp_dV2's), eqp_vs_U.txt,
E_vs_U.png (vs. the exact ED ground-state energy), eqp_vs_U.png.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scan_ed_deltaV"))
import make_plot as ed_dV  # scan_ed_deltaV/make_plot.py

# eqp_vs_U.png colors: one color per +-symmetric pair (outermost first), not per raw sorted
# eigenvalue index -- see scan_ng2_not_imp_dV2_U.py's branch_color for the full rationale.
PAIR_COLORS = ["tab:orange", "tab:blue"]
GHOST_COLOR = "0.6"


def branch_color(i, n):
    pair = min(i, n - 1 - i)
    return PAIR_COLORS[pair] if pair < len(PAIR_COLORS) else GHOST_COLOR

Ng, t, deltaV = 2, 1.0, 2.0
OUT = "scan_ng2_imp_dV2"


def run_branch(name, Us):
    rows = []
    R0 = R1 = lam0 = lam1 = None  # cold start at Us[0]
    for U in Us:
        d = os.path.join(OUT, name, f"U_{U:.1f}")
        os.makedirs(d, exist_ok=True)
        if R0 is not None:
            np.savetxt(os.path.join(d, "R.in"), np.vstack([R0, R1]),
                       header=f"input R guess for U/t={U:.1f} (previous point's converged R)")
            np.savetxt(os.path.join(d, "L.in"), np.vstack([lam0, lam1]),
                       header=f"input lambda guess for U/t={U:.1f} (previous point's converged lambda)")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = g.run_gga(Ng, U, t=t, deltaV=deltaV, Rg0=R0, Rg1=R1, lamg0=lam0, lamg1=lam1, verbose=True)

        with open(os.path.join(d, "output.out"), "w") as f:
            f.write(buf.getvalue())
            f.write(f"\nNg={Ng}  U/t={U:.4f}  deltaV/t={deltaV:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
                    f"E_var/t={res['E_var']/t:.8f}  docc={res['docc']:.6f}  "
                    f"converged={res['converged']}  iters={res['iters']}\n")
        g.save_results(res, prefix=os.path.join(d, f"gga_Ng{Ng}_U{U:g}"))

        R0, R1, lam0, lam1 = res["R0"], res["R1"], res["lam0"], res["lam1"]
        Hqp, rho = g.qp_state([R0, R1], [lam0, lam1], t)
        eqp = np.linalg.eigvalsh(Hqp)
        np.savetxt(os.path.join(d, "eqp.txt"), eqp,
                   header=f"quasiparticle energies (eigenvalues of H_qp), U/t={U:.4f}, deltaV/t={deltaV:.4f}")

        rows.append(dict(U=U, E=res["E_var"], Z0=res["Z0"], Z1=res["Z1"], docc=res["docc"],
                          conv=res["converged"], iters=res["iters"], eqp=eqp,
                          R0=R0, R1=R1, lam0=lam0, lam1=lam1))
        print(f"[{name:8s}] U/t={U:5.1f}  converged={res['converged']!s:5s}  iters={res['iters']:3d}  "
              f"E_var/t={res['E_var']/t: .8f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}", flush=True)
    return rows


def ed_poles(Us):
    """Exact (ED) pole positions (4 per U: +-omega_g, +-omega_u) at the correctly-converted
    deltaV_ed = 2*deltaV -- exact_dimer_deltaV.py's deltaV and gga_dimer.py's deltaV are NOT
    the same physical parameter, see scan_ng2_not_imp_dV2_U.py's ed_ground_state_energy
    docstring for the full explanation."""
    return np.array([sorted(ed_dV.analytic_poles(t, U, 2.0 * deltaV)[0]) for U in Us])


def write_branch(name, rows):
    rows = sorted(rows, key=lambda r: r["U"])
    d = os.path.join(OUT, name)
    U = np.array([r["U"] for r in rows])
    eq = np.array([r["eqp"] for r in rows])
    ed_p = ed_poles(U)
    np.savetxt(os.path.join(d, "eqp_vs_U.txt"), np.column_stack([U, eq]),
               header="U/t " + " ".join(f"eqp{i}" for i in range(eq.shape[1])))
    plt.figure(figsize=(6, 4.5))
    n = eq.shape[1]
    for i in range(n):
        plt.plot(U, eq[:, i], "-", lw=1.2, color=branch_color(i, n))
    for i in range(ed_p.shape[1]):
        plt.plot(U, ed_p[:, i], "--", lw=1.2, color="black",
                  label="exact (ED)" if i == 0 else None)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title(f"eqp vs U ({name}, Ng={Ng}, $\\Delta V/t={deltaV:g}$)")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(d, "eqp_vs_U.png"), dpi=150); plt.close()
    return rows


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "forward"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "backward"), exist_ok=True)
    Us_up = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]
    Us_down = [round(u, 1) for u in np.arange(10.0, 0.09, -0.1)]

    fwd = write_branch("forward", run_branch("forward", Us_up))
    back = write_branch("backward", run_branch("backward", Us_down))

    F = {r["U"]: r for r in fwd}
    B = {r["U"]: r for r in back}
    Us = sorted(set(F) & set(B))
    best = [F[u] if F[u]["E"] <= B[u]["E"] else B[u] for u in Us]
    which = ["forward" if F[u]["E"] <= B[u]["E"] else "backward" for u in Us]

    with open(os.path.join(OUT, "lowest_energy.txt"), "w") as f:
        f.write("# U/t  E_var/t  branch  eqp0..eqp3   (lowest-energy solution of the two scans)\n")
        for u, r, w in zip(Us, best, which):
            f.write(f"{u:5.1f} {r['E']: .8f} {w:8s} " + " ".join(f"{x: .6f}" for x in r["eqp"]) + "\n")
    U = np.array(Us)
    eq = np.array([r["eqp"] for r in best])
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.column_stack([U, eq]),
               header="U/t eqp0..eqp3 (lowest-energy solution)")

    # deltaV_ed = 2*deltaV, NOT deltaV: exact_dimer_deltaV.py's deltaV and gga_dimer.py's
    # deltaV are different physical parameters (see scan_ng2_not_imp_dV2_U.py's
    # ed_ground_state_energy docstring for the full explanation and how this was caught
    # -- gGA energy coming out BELOW the unconverted-deltaV ED energy, violating the
    # variational principle).
    E_ed = np.array([ed_dV.E0_explicit(t, u, 2.0 * deltaV) for u in U])
    E_gga = np.array([r["E"] for r in best])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, [F[u]["E"] for u in Us], "--", lw=1.2, color="tab:blue", label="forward (from U=0.1)")
    ax[0].plot(U, [B[u]["E"] for u in Us], "-.", lw=1.2, color="tab:green", label="backward (from U=10)")
    ax[0].plot(U, E_gga, "-", lw=1.8, color="black", label=f"gGA, Ng={Ng}, lowest of the two")
    ax[0].plot(U, E_ed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("E_var / t")
    ax[0].set_title(f"Variational energy vs U (Ng={Ng}, gga_dimer.py, $\\Delta V/t={deltaV:g}$)")
    ax[0].legend(fontsize=8)
    ax[1].plot(U, E_gga - E_ed, "-", lw=1.2, color="tab:purple")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("gGA - ED")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "E_vs_U.png"), dpi=150); plt.close()

    ed_p = ed_poles(U)
    n = eq.shape[1]
    plt.figure(figsize=(6.4, 4.8))
    for i in range(n):
        plt.plot(U, eq[:, i], "-", lw=1.2, color=branch_color(i, n))
    for i in range(ed_p.shape[1]):
        plt.plot(U, ed_p[:, i], "--", lw=1.2, color="black",
                  label="exact (ED)" if i == 0 else None)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title(f"Quasiparticle energies, lowest-energy solution (Ng={Ng}, $\\Delta V/t={deltaV:g}$)")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150); plt.close()

    from collections import Counter
    print("\nlowest-energy branch counts:", dict(Counter(which)))
    print(f"total {time.time()-t0:.0f}s; wrote {OUT}/")
