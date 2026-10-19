"""U-continuation scans (Ng=4, Delta V/t=2.0) with gga_dimer.py's iterative embedding --
the Ng=4 analogue of scan_ng2_imp_dV2_U.py.

Two branches, both by continuation in U (every U uses the previous U's converged (R,
lambda) as the next guess):

  forward   U/t = 0.1 -> 10.0 in steps of +0.1.
  backward  U/t = 10.0 -> 0.1 in steps of -0.1.

Unlike scan_ng2_imp_dV2_U.py, each branch does NOT start cold: a cold start at Ng=4 risks
landing on the trivial "sleeping ghost" fixed point (confirmed to happen for the Ng=2
script's own backward branch at deltaV=2). Instead each branch is seeded, at its own first
U (0.1 for forward, 10.0 for backward), from the already-converged, independently-obtained
not_imp_gga_dimer.py Ng=4 solution at that U (scan_ng4_not_imp_dV2/U_<U>/), exactly the
seeding already validated point-by-point (U=8.8, 9.5, 5.0: agreement to 1e-6..1e-9 in
E_var and the spectral function) before committing to a full scan. Only the FIRST point of
each branch is seeded this way; every subsequent U continues from gga_dimer's own previous
converged point, so each branch can still explore/settle into whatever fixed point its own
dynamics prefers along the way -- which is exactly why both directions are kept and the
lower-energy one retained at every U (same reasoning as scan_ng2_imp_dV2_U.py: likely to
have genuinely different forward/backward solutions at some U).

For every U it writes scan_ng4_imp_dV2/<branch>/U_<U>/ : R.in, L.in (the guess fed in),
output.out (run_gga trace + summary), gga_Ng4_U*_{R,lambda,lambda_c}.dat, eqp.txt. Per
branch: eqp_vs_U.txt, eqp_vs_U.png. Combined, directly in scan_ng4_imp_dV2/:
lowest_energy.txt, eqp_vs_U.txt, E_vs_U.png (vs. not_imp's Ng=4 scan and the exact ED
ground-state energy), eqp_vs_U.png.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scan_ed_deltaV"))
import make_plot as ed_dV  # scan_ed_deltaV/make_plot.py

# eqp_vs_U.png colors: one color per +-symmetric pair (outermost first) -- see
# scan_ng2_not_imp_dV2_U.py's branch_color for the full rationale.
PAIR_COLORS = ["tab:orange", "tab:blue"]
GHOST_COLOR = "0.6"


def branch_color(i, n):
    pair = min(i, n - 1 - i)
    return PAIR_COLORS[pair] if pair < len(PAIR_COLORS) else GHOST_COLOR


Ng, t, deltaV = 4, 1.0, 2.0
OUT = "scan_ng4_imp_dV2"
SC_NOT_IMP = "scan_ng4_not_imp_dV2"


def not_imp_seed(U):
    d = os.path.join(SC_NOT_IMP, f"U_{U:.1f}")
    R = np.loadtxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_R.dat"))
    lam = np.loadtxt(os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}_lambda.dat"))
    N = R.shape[1]
    return R[0], R[1], lam[:N], lam[N:]


def run_branch(name, Us):
    rows = []
    R0, R1, lam0, lam1 = not_imp_seed(Us[0])
    for U in Us:
        d = os.path.join(OUT, name, f"U_{U:.1f}")
        os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "R.in"), np.vstack([R0, R1]),
                   header=f"input R guess for U/t={U:.1f}")
        np.savetxt(os.path.join(d, "L.in"), np.vstack([lam0, lam1]),
                   header=f"input lambda guess for U/t={U:.1f}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = g.run_gga(Ng, U, t=t, deltaV=deltaV, tol_E=1e-10, tol_mat=1e-10, max_iter=300,
                             Rg0=R0, Rg1=R1, lamg0=lam0, lamg1=lam1, verbose=True)

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


def refine_with_notimp_seed(Us, best, which, tol=1e-6):
    """Patch any U where gga_dimer's own forward/backward continuation got stuck on a
    higher-energy fixed point (found: a transition region U/t~6.8-7.7, right before the
    large-U peak-splitting onset around U/t~8 -- classic continuation hysteresis around a
    level crossing) by reseeding directly from not_imp_gga_dimer.py's own Ng=4 result at
    that U (already validated point-by-point: agreement to 1e-6..1e-9 away from this
    region) and keeping it if it's lower. Patches whichever branch is the current winner's
    own saved files, same as scan_ng2_not_imp_dV2_U.py's refine_with_gga_seed (there the
    roles were reversed: not_imp was refined from gga; here gga is refined from not_imp)."""
    n_improved = 0
    for i, U in enumerate(Us):
        R0, R1, lam0, lam1 = not_imp_seed(U)
        res = g.run_gga(Ng, U, t=t, deltaV=deltaV, tol_E=1e-12, tol_mat=1e-12, max_iter=500,
                         Rg0=R0, Rg1=R1, lamg0=lam0, lamg1=lam1, verbose=False)
        if res["converged"] and res["E_var"] < best[i]["E"] - tol:
            d = os.path.join(OUT, which[i], f"U_{U:.1f}")
            g.save_results(res, prefix=os.path.join(d, f"gga_Ng{Ng}_U{U:g}"))
            with open(os.path.join(d, "output.out"), "w") as f:
                f.write(f"Ng={Ng}  U/t={U:.4f}  deltaV/t={deltaV:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
                        f"E_var/t={res['E_var']/t:.8f}  docc={res['docc']:.6f}  "
                        f"converged={res['converged']}  iters={res['iters']}  (refined from not_imp seed)\n")
            Hqp, rho = g.qp_state([res["R0"], res["R1"]], [res["lam0"], res["lam1"]], t)
            eqp = np.linalg.eigvalsh(Hqp)
            np.savetxt(os.path.join(d, "eqp.txt"), eqp,
                       header=f"quasiparticle energies, U/t={U:.4f}, deltaV/t={deltaV:.4f} (refined)")
            print(f"[refine ] U/t={U:5.1f}  gga: {best[i]['E']: .8f} -> {res['E_var']: .8f}  "
                  f"(branch={which[i]})", flush=True)
            best[i] = dict(U=U, E=res["E_var"], Z0=res["Z0"], Z1=res["Z1"], docc=res["docc"],
                            conv=res["converged"], iters=res["iters"], eqp=eqp,
                            R0=res["R0"], R1=res["R1"], lam0=res["lam0"], lam1=res["lam1"])
            n_improved += 1
    print(f"\nrefine pass: {n_improved}/{len(Us)} points improved by seeding from not_imp_gga_dimer.py", flush=True)
    return best


def ed_poles(Us):
    """deltaV_ed = 2*deltaV -- see scan_ng2_not_imp_dV2_U.py's ed_ground_state_energy
    docstring for why exact_dimer_deltaV.py's deltaV is not the same parameter."""
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

    best = refine_with_notimp_seed(Us, best, which)

    with open(os.path.join(OUT, "lowest_energy.txt"), "w") as f:
        f.write("# U/t  E_var/t  branch  eqp0..eqp9   (lowest-energy solution of the two scans)\n")
        for u, r, w in zip(Us, best, which):
            f.write(f"{u:5.1f} {r['E']: .8f} {w:8s} " + " ".join(f"{x: .6f}" for x in r["eqp"]) + "\n")
    U = np.array(Us)
    eq = np.array([r["eqp"] for r in best])
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.column_stack([U, eq]),
               header="U/t eqp0..eqp9 (lowest-energy solution)")

    # not_imp's own Ng=4 lowest-energy result, for comparison
    import re
    E_notimp = []
    for u in Us:
        txt = open(os.path.join(SC_NOT_IMP, f"U_{u:.1f}", "output.out")).read()
        E_notimp.append(float(re.search(r"E_var/t=(-?[\d.]+)", txt).group(1)))
    E_notimp = np.array(E_notimp)

    E_ed = np.array([ed_dV.E0_explicit(t, u, 2.0 * deltaV) for u in U])
    E_gga = np.array([r["E"] for r in best])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, [F[u]["E"] for u in Us], "--", lw=1.2, color="tab:blue", label="forward (from U=0.1)")
    ax[0].plot(U, [B[u]["E"] for u in Us], "-.", lw=1.2, color="tab:green", label="backward (from U=10)")
    ax[0].plot(U, E_gga, "-", lw=1.8, color="black", label=f"gGA, Ng={Ng}, lowest of the two")
    ax[0].plot(U, E_notimp, ":", lw=1.2, color="tab:brown", label="not_imp_gga_dimer, Ng=4")
    ax[0].plot(U, E_ed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("E_var / t")
    ax[0].set_title(f"Variational energy vs U (Ng={Ng}, gga_dimer.py, $\\Delta V/t={deltaV:g}$)")
    ax[0].legend(fontsize=8)
    ax[1].plot(U, E_gga - E_ed, "-", lw=1.2, color="tab:purple", label="gga-ED")
    ax[1].plot(U, E_gga - E_notimp, "-", lw=1.2, color="tab:red", label="gga-not_imp")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("energy difference"); ax[1].legend(fontsize=7)
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
    print(f"max|E_gga-E_notimp| = {np.abs(E_gga-E_notimp).max():.3e}")
    print(f"total {time.time()-t0:.0f}s; wrote {OUT}/")
