"""U-continuation scans (Ng=2) with the direct minimization of not_imp_gga_dimer.py.

Two branches, both by continuation in U (every U is started from the previous U's
converged variables z = (x0, x1, X); the constraints do not depend on U, so the
previous solution is exactly feasible at the new U):

  backward  U/t = 10.0 -> 0.1 in steps of -0.1, starting from the U=10 guess
            (examples/U10/R.in, L.in) through start_from_R_lambda.
  forward   U/t = 0.1 -> 10.0 in steps of +0.1. The starting point at U=0.1 is
            obtained by priming (not saved): the U=2 forward seed
            (examples/U2_forward) is converged at U=2 and continued down to 0.1.
            In the direct minimization this solution (inert ghosts) satisfies the
            constraints and the KKT conditions, so it is a legitimate start.

Every point is minimized until the constraint violation is < 1e-9 and the
gradient projected on the constraint tangent space is < 5e-6 (up to 5 SLSQP passes).

For every U it writes scan_not_imp/<branch>/U_<U>/ : output.out (one-line summary),
not_imp_Ng2_U*_{R,lambda,lambda_c}.dat, eqp.txt (eigenvalues of H_qp(R, lambda)),
z.npy (the variables, to restart). Per branch: summary.txt (U, E_var, Z0, Z1, docc,
residuals), eqp_vs_U.txt, and the plots E_vs_U.png, eqp_vs_U.png.
Combined, directly in scan_not_imp/: E_vs_U.png (both branches, their lower envelope
and, for reference, the lower envelope of the gga_dimer.py scans in scan/), eqp_vs_U.png
and eqp_vs_U.txt (quasiparticle energies of the lowest-energy solution at each U).
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

Ng, t = 2, 1.0
OUT = "scan_not_imp"


def tangent_grad(model, z):
    E, gE = model.energy(z)
    J = model.constraint_jac(z)
    u, s, vt = np.linalg.svd(J)
    rank = int((s > 1e-9 * s[0]).sum())
    return np.linalg.norm(vt[rank:] @ gE)


def converge(model, z, max_pass=5):
    """SLSQP passes until feasible and stationary; returns (z, passes, |c|, tangent gradient)."""
    for k in range(1, max_pass + 1):
        z = model.minimize(z, max_iter=3000).x
        viol = np.abs(model.constraints(z)).max()
        tg = tangent_grad(model, z)
        if viol < 1e-9 and tg < 5e-6:
            break
    return z, k, viol, tg


def solve_point(U, z_start):
    model = ni.NotImpDimer(Ng, U, t)
    z, npass, viol, tg = converge(model, z_start)
    return model, z, npass, viol, tg


def save_point(d, model, z, npass, viol, tg):
    os.makedirs(d, exist_ok=True)
    a = model.analyse(z)
    with contextlib.redirect_stdout(io.StringIO()):
        g.save_results(a, prefix=os.path.join(d, f"not_imp_Ng{Ng}_U{model.U:g}"))
    np.savetxt(os.path.join(d, "eqp.txt"), a["eqp"],
               header=f"quasiparticle energies (eigenvalues of H_qp), U/t={model.U:.4f}")
    np.save(os.path.join(d, "z.npy"), z)
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(f"Ng={Ng}  U/t={model.U:.4f}  Z0={a['Z0']:.6f}  Z1={a['Z1']:.6f}  "
                f"E_var/t={a['E_var']:.8f}  docc={a['docc']:.6f}  passes={npass}  "
                f"|c|={viol:.2e}  |tangent grad|={tg:.2e}  KKT(phi,psi,imp)="
                f"{a['kkt_phi']:.1e},{a['kkt_psi']:.1e},{a['kkt_imp']:.1e}\n")
    return a


def run_branch(name, Us, z_start, model_at=None, prime=None):
    """prime: optional list of U values traversed first WITHOUT saving."""
    z = z_start
    if prime is not None:
        for U in prime:
            _, z, npass, viol, tg = solve_point(U, z)
            if viol > 1e-8:
                print(f"  [{name} prime] WARNING infeasible at U={U}: |c|={viol:.1e}")
    rows = []
    for U in Us:
        model, z, npass, viol, tg = solve_point(U, z)
        d = os.path.join(OUT, name, f"U_{U:.1f}")
        a = save_point(d, model, z, npass, viol, tg)
        rows.append(dict(U=U, E=a["E_var"], Z0=a["Z0"], Z1=a["Z1"], docc=a["docc"], viol=viol,
                         tg=tg, kkt_phi=a["kkt_phi"], kkt_psi=a["kkt_psi"], eqp=a["eqp"]))
        flag = "" if (viol < 1e-9 and tg < 5e-6) else "   <-- not fully converged"
        print(f"[{name:8s}] U/t={U:5.1f}  E_var/t={a['E_var']: .8f}  Z0={a['Z0']:.6f}  Z1={a['Z1']:.6f}  "
              f"passes={npass}  |c|={viol:.1e}  tg={tg:.1e}{flag}", flush=True)
    return rows


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
    plt.figure(figsize=(6, 4.5))
    plt.plot(U, [r["E"] for r in rows], "-", lw=1.5)
    plt.xlabel("U/t"); plt.ylabel("E_var / t"); plt.title(f"E_var vs U ({name}, direct minimization, Ng={Ng})")
    plt.tight_layout(); plt.savefig(os.path.join(d, "E_vs_U.png"), dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5))
    eq = np.array([r["eqp"] for r in rows])
    for i in range(eq.shape[1]):
        plt.plot(U, eq[:, i], "-", lw=1.2)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title(f"eqp vs U ({name}, direct minimization, Ng={Ng})")
    plt.tight_layout(); plt.savefig(os.path.join(d, "eqp_vs_U.png"), dpi=150); plt.close()
    return rows


def gga_dimer_envelope():
    """Lower envelope of the E_var of the two gga_dimer.py scans (scan/backward, scan/forward)."""
    import glob, re
    E = {}
    for branch in ("backward", "forward"):
        for f in glob.glob(os.path.join("scan", branch, "U_*", "output.out")):
            m = re.search(r"U/t=([\d.]+).*?E_var/t=(-?[\d.]+)", open(f).readlines()[-1])
            u, e = round(float(m.group(1)), 1), float(m.group(2))
            E[u] = min(E.get(u, np.inf), e)
    us = sorted(E)
    return np.array(us), np.array([E[u] for u in us])


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    Us_down = [round(u, 1) for u in np.arange(10.0, 0.09, -0.1)]
    Us_up = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]

    # ---- backward: from the U=10 guess downward -------------------------------
    R = list(g.read_R_guess(os.path.join("examples", "U10", "R.in")))
    lam = list(g.read_lambda_guess(os.path.join("examples", "U10", "L.in")))
    z0 = ni.NotImpDimer(Ng, 10.0, t).start_from_R_lambda(R, lam)
    back = write_branch("backward", run_branch("backward", Us_down, z0))

    # ---- forward: U=2 forward seed, primed down to 0.1, then upward -------------
    R = list(g.read_R_guess(os.path.join("examples", "U2_forward", "R.in")))
    lam = list(g.read_lambda_guess(os.path.join("examples", "U2_forward", "L.in")))
    z0 = ni.NotImpDimer(Ng, 2.0, t).start_from_R_lambda(R, lam)
    prime = [round(u, 1) for u in np.arange(2.0, 0.09, -0.1)]
    fwd = write_branch("forward", run_branch("forward", Us_up, z0, prime=prime))

    # ---- combined plots: lowest-energy solution at each U --------------------------
    B = {r["U"]: r for r in back}
    F = {r["U"]: r for r in fwd}
    Us = sorted(set(B) & set(F))
    best = [B[u] if B[u]["E"] <= F[u]["E"] else F[u] for u in Us]
    which = ["backward" if B[u]["E"] <= F[u]["E"] else "forward" for u in Us]
    with open(os.path.join(OUT, "lowest_energy.txt"), "w") as f:
        f.write("# U/t  E_var/t  branch  eqp0..eqp5   (lowest-energy solution of the two scans)\n")
        for u, r, w in zip(Us, best, which):
            f.write(f"{u:5.1f} {r['E']: .8f} {w:8s} " + " ".join(f"{x: .6f}" for x in r["eqp"]) + "\n")
    np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), np.array([[u] + r["eqp"].tolist() for u, r in zip(Us, best)]),
               header="U/t eqp0..eqp5 (lowest-energy solution)")

    ug, eg = gga_dimer_envelope()
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(Us, [B[u]["E"] for u in Us], "-", lw=1.5, label="direct min., backward (from U=10)")
    plt.plot(Us, [F[u]["E"] for u in Us], "--", lw=1.5, label="direct min., forward (from U=0.1)")
    plt.plot(Us, [r["E"] for r in best], ":", lw=2.2, color="black", label="lowest of the two")
    plt.plot(ug, eg, "-", lw=0.8, color="tab:red", alpha=0.7, label="gga_dimer.py, lowest of its 2 scans")
    plt.xlabel("U/t"); plt.ylabel("E_var / t"); plt.title(f"Variational energy vs U (Ng={Ng})")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E_vs_U.png"), dpi=150); plt.close()

    eq = np.array([r["eqp"] for r in best])
    plt.figure(figsize=(6.4, 4.8))
    for i in range(eq.shape[1]):
        plt.plot(Us, eq[:, i], "-", lw=1.2)
    plt.axhline(0.0, color="gray", lw=0.5, ls=":")
    plt.xlabel("U/t"); plt.ylabel("quasiparticle energy / t")
    plt.title(f"Quasiparticle energies, lowest-energy solution (Ng={Ng})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150); plt.close()

    from collections import Counter
    print("\nlowest-energy branch counts:", dict(Counter(which)))
    print(f"total time {time.time() - t0:.0f}s; wrote {OUT}/")
