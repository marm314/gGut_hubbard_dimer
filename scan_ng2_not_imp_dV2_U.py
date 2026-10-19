"""U-continuation scans (Ng=2, Delta V/t=2.0) with the direct minimization of
not_imp_gga_dimer.py -- the Delta V=2 analogue of scan_not_imp_U.py, kept as a SEPARATE
script/output folder so the existing Delta V=0 results are untouched.

Two branches, both by continuation in U (every U is started from the previous U's
converged variables z = (x0, x1, X); the constraints do not depend on U, so the previous
solution is exactly feasible at the new U), exactly as in scan_not_imp_U.py -- a SINGLE
forward-only branch was tried first and found to land in a higher-energy, suboptimal fixed
point at some U (cross-checked against gga_dimer.py seeded at the same point, which found
a notably lower energy at U/t=5, 8), so both directions are needed here too:

  forward   U/t = 0.1 -> 10.0 in steps of +0.1. At U=0.1, NSTARTS random starts of
            not_imp_gga_dimer.NotImpDimer(2, 0.1, t, deltaV=2.0) are minimized and the
            lowest feasible one kept.
  backward  U/t = 10.0 -> 0.1 in steps of -0.1. At U=10.0, likewise NSTARTS random starts.

If a continued point fails to meet the convergence criteria after 5 SLSQP passes, a few
extra random restarts at that same U are tried as a fallback. At every U, the lower-energy
of the two branches is kept as the reported solution (deltaV=2 breaks the Z0=Z1
site-exchange symmetry of the deltaV=0 problem, hence the extra exploration).

For every U it writes scan_ng2_not_imp_dV2/<branch>/U_<U>/ : output.out (one-line summary),
not_imp_Ng2_U*_{R,lambda,lambda_c}.dat, eqp.txt (eigenvalues of H_qp(R, lambda)), z.npy
(the variables, to restart) -- needed so scan_ng4_not_imp_dV2_U.py can embed from the
winning branch exactly like scan_ng4_U.py embeds from scan_ng2_not_imp/. Per branch:
summary.txt, eqp_vs_U.txt, E_vs_U.png, eqp_vs_U.png. Combined, directly in
scan_ng2_not_imp_dV2/: lowest_energy.txt (same format as scan_not_imp_U.py's, branch =
whichever won at that U, for scan_ng4_not_imp_dV2_U.py to read), eqp_vs_U.txt, and
E_vs_U.png/eqp_vs_U.png overlaid with the exact (ED) ground-state energy at this deltaV
(scan_ed_deltaV/ED_deltaV.tex's closed form) for comparison.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scan_ed_deltaV"))
import make_plot as ed_dV  # scan_ed_deltaV/make_plot.py

# eqp_vs_U.png colors: one color per +-symmetric pair (outermost pair first), not per raw
# sorted eigenvalue index -- a plain default color cycle over 6 (Ng=2) sorted eigenvalues
# gives a meaningless rainbow since the sort order of a near-symmetric spectrum jumps
# around at level crossings. Deeper ("ghost") pairs beyond the outer two, which have no
# counterpart among ED's 4 exact poles, are all drawn in the same muted gray.
PAIR_COLORS = ["tab:orange", "tab:blue"]
GHOST_COLOR = "0.6"


def branch_color(i, n):
    pair = min(i, n - 1 - i)  # 0 = outermost +- pair, 1 = next, ...
    return PAIR_COLORS[pair] if pair < len(PAIR_COLORS) else GHOST_COLOR

Ng, t, deltaV = 2, 1.0, 2.0
OUT = "scan_ng2_not_imp_dV2"
NSTARTS = 12


def tangent_grad(model, z):
    E, gE = model.energy(z)
    J = model.constraint_jac(z)
    u, s, vt = np.linalg.svd(J)
    rank = int((s > 1e-9 * s[0]).sum())
    return np.linalg.norm(vt[rank:] @ gE)


def converge(model, z, max_pass=5):
    for k in range(1, max_pass + 1):
        z = model.minimize(z, max_iter=3000).x
        viol = np.abs(model.constraints(z)).max()
        tg = tangent_grad(model, z)
        if viol < 1e-9 and tg < 5e-6:
            break
    return z, k, viol, tg


def ok(viol, tg):
    return viol < 1e-9 and tg < 5e-6


def first_point(U, seed):
    """NSTARTS random starts, keep the lowest feasible."""
    model = ni.NotImpDimer(Ng, U, t, deltaV=deltaV)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(NSTARTS):
        z0 = model.random_start(rng)
        z, npass, viol, tg = converge(model, z0)
        if viol < 1e-8 and (best is None or model.energy(z, grad=False) < best[0] - 1e-12):
            best = (model.energy(z, grad=False), z, npass, viol, tg)
    return model, best


def continued_point(U, z_prev, seed):
    """Continue from z_prev; a few random restarts as a fallback if that doesn't converge."""
    model = ni.NotImpDimer(Ng, U, t, deltaV=deltaV)
    z, npass, viol, tg = converge(model, z_prev)
    best = (model.energy(z, grad=False), z, npass, viol, tg)
    if ok(viol, tg):
        return model, best
    rng = np.random.default_rng(seed)
    for _ in range(6):
        z0 = model.random_start(rng)
        zc, npass2, viol2, tg2 = converge(model, z0)
        if viol2 < 1e-8 and model.energy(zc, grad=False) < best[0] - 1e-12:
            best = (model.energy(zc, grad=False), zc, npass2, viol2, tg2)
            if ok(viol2, tg2):
                break
    return model, best


def save_point(name, U, model, best):
    E, z, npass, viol, tg = best
    a = model.analyse(z)
    d = os.path.join(OUT, name, f"U_{U:.1f}")
    os.makedirs(d, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        g.save_results(a, prefix=os.path.join(d, f"not_imp_Ng{Ng}_U{U:g}"))
    np.savetxt(os.path.join(d, "eqp.txt"), a["eqp"],
               header=f"quasiparticle energies (eigenvalues of H_qp), U/t={U:.4f}, deltaV/t={deltaV:.4f}")
    np.save(os.path.join(d, "z.npy"), z)
    conv = ok(viol, tg)
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(f"Ng={Ng}  U/t={U:.4f}  deltaV/t={deltaV:.4f}  Z0={a['Z0']:.6f}  Z1={a['Z1']:.6f}  "
                f"E_var/t={a['E_var']:.8f}  docc={a['docc']:.6f}  passes={npass}  "
                f"|c|={viol:.2e}  |tangent grad|={tg:.2e}  converged={conv}  KKT(phi,psi,imp)="
                f"{a['kkt_phi']:.1e},{a['kkt_psi']:.1e},{a['kkt_imp']:.1e}\n")
    return dict(U=U, E=a["E_var"], Z0=a["Z0"], Z1=a["Z1"], docc=a["docc"], viol=viol, tg=tg,
                kkt_phi=a["kkt_phi"], kkt_psi=a["kkt_psi"], eqp=a["eqp"], conv=conv)


def run_branch(name, Us, seed0):
    """Us: the full ordered list of U values for this branch (first entry seeded by
    random multi-start, the rest by continuation)."""
    rows = []
    model, best = first_point(Us[0], seed0)
    r = save_point(name, Us[0], model, best)
    rows.append(r)
    print(f"[{name:8s}] U/t={Us[0]:5.1f}  E_var/t={r['E']: .8f}  Z0={r['Z0']:.6f}  Z1={r['Z1']:.6f}  "
          f"|c|={r['viol']:.1e}  tg={r['tg']:.1e}{'' if r['conv'] else '  <-- not fully converged'}", flush=True)

    z_prev = best[1]
    for U in Us[1:]:
        model, best = continued_point(U, z_prev, seed=seed0 + int(round(U * 10)))
        z_prev = best[1]
        r = save_point(name, U, model, best)
        rows.append(r)
        print(f"[{name:8s}] U/t={U:5.1f}  E_var/t={r['E']: .8f}  Z0={r['Z0']:.6f}  Z1={r['Z1']:.6f}  "
              f"passes={best[2]}  |c|={r['viol']:.1e}  tg={r['tg']:.1e}"
              f"{'' if r['conv'] else '  <-- not fully converged'}", flush=True)
    return rows


def refine_with_gga_seed(Us, best, which, gga_dir="scan_ng2_imp_dV2"):
    """Cross-check/refine against gga_dimer.py's own forward+backward scan (scan_ng2_imp_dV2_U.py):
    for every U, seed not_imp_gga_dimer from gga_dimer's winning (R, lambda) at that U
    (via start_from_R_lambda) and converge it the same way as the native branches. This
    caught a real bug: the native forward+backward scan above still landed in a higher-energy
    fixed point at several U (e.g. U/t=5, 8), which gga_dimer.py's independent iterative
    embedding did not -- comparing the two solvers, not just trusting either one's own
    convergence flags, is what exposed it. Patches the winning not_imp branch's own saved
    files at any U where this seed does strictly better, so the per-branch records stay
    consistent with the reported best."""
    gga_lowest = {}
    for line in open(os.path.join(gga_dir, "lowest_energy.txt")):
        if not line.startswith("#") and line.strip():
            p = line.split()
            gga_lowest[round(float(p[0]), 1)] = p[2]

    n_improved = 0
    for i, U in enumerate(Us):
        branch = gga_lowest.get(U)
        if branch is None:
            continue
        d = os.path.join(gga_dir, branch, f"U_{U:.1f}")
        R = np.loadtxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_R.dat"))
        lam = np.loadtxt(os.path.join(d, f"gga_Ng{Ng}_U{U:g}_lambda.dat"))
        N = R.shape[1]
        model = ni.NotImpDimer(Ng, U, t, deltaV=deltaV)
        z0 = model.start_from_R_lambda([R[0], R[1]], [lam[:N], lam[N:]])
        z, npass, viol, tg = converge(model, z0)
        E = model.energy(z, grad=False)
        if viol < 1e-8 and E < best[i]["E"] - 1e-9:
            r = save_point(which[i], U, model, (E, z, npass, viol, tg))
            print(f"[refine ] U/t={U:5.1f}  gga-seeded not_imp: {best[i]['E']: .8f} -> {E: .8f}  "
                  f"(branch={which[i]})", flush=True)
            best[i] = r
            n_improved += 1
    print(f"\nrefine pass: {n_improved}/{len(Us)} points improved by seeding from gga_dimer.py", flush=True)
    return best


def ed_ground_state_energy(Us):
    """Exact (ED) ground-state energy at this deltaV, from the already-validated closed
    form of scan_ed_deltaV/ED_deltaV.tex (E0_explicit, cross-checked there to machine
    precision against exact_dimer_deltaV.py's numerical ED) -- the physically meaningful
    reference for the gGA variational energy (a deltaV=0 gGA curve is not, since deltaV
    changes the actual ground state being approximated).

    IMPORTANT: exact_dimer_deltaV.py's deltaV and gga_dimer.py's deltaV are NOT the same
    physical parameter -- exact_dimer_deltaV.py's one-body term is -mu(n0+n1)+(deltaV/2)(n0-n1)
    with mu=U/2 (site-0 coefficient -U/2+deltaV/2), while gga_dimer.py's (C++-validated)
    convention is mu_0=U/2+deltaV, mu_1=U/2-deltaV (site-0 coefficient -U/2-deltaV). Matching
    coefficients gives deltaV_ed = -2*deltaV_gga; since E0 is an even function of deltaV (the
    site-exchange symmetry of the model), deltaV_ed = +2*deltaV_gga works just as well and is
    used here. Getting this wrong makes the gGA energy appear to violate the variational
    principle (found by comparing against the UNconverted deltaV=2 ED energy: gGA came out
    BELOW it for U/t up to ~5.5, by as much as 1.6 -- impossible for a genuine variational
    upper bound, and the tell that the two scripts' deltaV did not mean the same thing).
    Confirmed fixed by this factor: gGA - ED >= 0 (down to ~0, at U=0.1) for every U tried."""
    return np.array([ed_dV.E0_explicit(t, U, 2.0 * deltaV) for U in Us])


def ed_poles(Us):
    """Exact (ED) pole positions (4 per U: +-omega_g, +-omega_u) at the correctly-converted
    deltaV_ed = 2*deltaV (see ed_ground_state_energy above), from the same closed form --
    the physically meaningful reference for the eqp_vs_U.png quasiparticle-energy plots,
    overlaid the same way (red dashed, "exact (ED)") as on E_vs_U.png."""
    return np.array([sorted(ed_dV.analytic_poles(t, U, 2.0 * deltaV)[0]) for U in Us])


def write_branch(name, rows):
    rows = sorted(rows, key=lambda r: r["U"])
    d = os.path.join(OUT, name)
    np.savetxt(os.path.join(d, "summary.txt"),
               np.array([[r["U"], r["E"], r["Z0"], r["Z1"], r["docc"], r["viol"], r["tg"],
                          r["kkt_phi"], r["kkt_psi"]] for r in rows]),
               header="U/t E_var/t Z0 Z1 docc max|c| |tangent grad| KKT_phi KKT_psi")
    np.savetxt(os.path.join(d, "eqp_vs_U.txt"), np.array([[r["U"]] + r["eqp"].tolist() for r in rows]),
               header="U/t eqp0..eqp5")
    U = np.array([r["U"] for r in rows])
    E_ed = ed_ground_state_energy(U)
    E_gga = np.array([r["E"] for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(6, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, E_gga, "-", lw=1.5, color="tab:blue", label=f"gGA, Ng={Ng}")
    ax[0].plot(U, E_ed, "--", lw=1.2, color="black", label="exact (ED)")
    ax[0].set_ylabel("E_var / t")
    ax[0].set_title(f"E_var vs U ({name}, Ng={Ng}, $\\Delta V/t={deltaV:g}$)")
    ax[0].legend(fontsize=8)
    ax[1].plot(U, E_gga - E_ed, "-", lw=1.2, color="tab:purple")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("gGA - ED")
    plt.tight_layout(); plt.savefig(os.path.join(d, "E_vs_U.png"), dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5))
    eq = np.array([r["eqp"] for r in rows])
    ed_p = ed_poles(U)
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

    fwd = write_branch("forward", run_branch("forward", Us_up, seed0=0))
    back = write_branch("backward", run_branch("backward", Us_down, seed0=10000))

    # ---- combined (top-level) outputs: lowest-energy of the two branches at each U -----
    F = {r["U"]: r for r in fwd}
    B = {r["U"]: r for r in back}
    Us = sorted(set(F) & set(B))
    best = [F[u] if F[u]["E"] <= B[u]["E"] else B[u] for u in Us]
    which = ["forward" if F[u]["E"] <= B[u]["E"] else "backward" for u in Us]

    if os.path.isdir("scan_ng2_imp_dV2"):
        best = refine_with_gga_seed(Us, best, which)

    with open(os.path.join(OUT, "lowest_energy.txt"), "w") as f:
        f.write("# U/t  E_var/t  branch  eqp0..eqp5   (lowest-energy solution of the two scans)\n")
        for u, r, w in zip(Us, best, which):
            f.write(f"{u:5.1f} {r['E']: .8f} {w:8s} " + " ".join(f"{x: .6f}" for x in r["eqp"]) + "\n")
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.array([[u] + r["eqp"].tolist() for u, r in zip(Us, best)]),
               header="U/t eqp0..eqp5 (lowest-energy solution)")

    U = np.array(Us)
    E_ed = ed_ground_state_energy(U)
    E_gga = np.array([r["E"] for r in best])
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax[0].plot(U, [F[u]["E"] for u in Us], "--", lw=1.2, color="tab:blue", label="forward (from U=0.1)")
    ax[0].plot(U, [B[u]["E"] for u in Us], "-.", lw=1.2, color="tab:green", label="backward (from U=10)")
    ax[0].plot(U, E_gga, "-", lw=1.8, color="black", label=f"gGA, Ng={Ng}, lowest of the two")
    ax[0].plot(U, E_ed, "--", lw=1.4, color="black", label="exact (ED)")
    ax[0].set_ylabel("E_var / t")
    ax[0].set_title(f"Variational energy vs U (Ng={Ng}, $\\Delta V/t={deltaV:g}$)")
    ax[0].legend(fontsize=8)
    ax[1].plot(U, E_gga - E_ed, "-", lw=1.2, color="tab:purple")
    ax[1].axhline(0.0, color="gray", lw=0.5, ls=":")
    ax[1].set_xlabel("U/t"); ax[1].set_ylabel("gGA - ED")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E_vs_U.png"), dpi=150); plt.close()

    eq = np.array([r["eqp"] for r in best])
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
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150); plt.close()

    from collections import Counter
    nb = sum(1 for r in fwd + back if not r["conv"])
    print(f"\nlowest-energy branch counts: {dict(Counter(which))}")
    print(f"points not fully converged (either branch): {nb};  total {time.time()-t0:.0f}s; wrote {OUT}/")
