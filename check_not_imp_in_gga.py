"""Are the lowest-energy solutions of the direct minimization (scan_not_imp/) also
solutions of the iterative embedding of gga_dimer.py?

For every U of scan_not_imp/lowest_energy.txt the (R, lambda) reconstructed from the
solution (not_imp_gga_dimer.analyse -> not_imp_Ng2_U*_R.dat, _lambda.dat) is used as the
seed of gga_dimer, and two things are measured.

(1) ONE embedding iteration from the seed (the cycle of gga_dimer.run_gga, iteration 0,
    plain substitution): dR = sum_I |R_I^new - R_I|, dDelta = max|Delta_qp - Delta_new|,
    the energy of the seed, and drho = max|rho_qp(R,lambda) - rho_solution| (does H_qp built
    from the reconstructed (R,lambda) reproduce the density matrix of the solution?).
    For a fixed point of the iteration dR, dDelta -> 0.
(2) A full run_gga from the seed (default tolerances): converged?, iterations, final E_var
    and its difference from the direct-minimization energy.

Writes scan_mix/gga_dimer_check.txt and scan_mix/gga_dimer_check.png.
"""
import contextlib
import glob
import io
import os

import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gga_dimer as g
import not_imp_gga_dimer as ni

Ng, t = 2, 1.0
N = Ng + 1
SC = "scan_not_imp"
OUT = "scan_mix"
os.makedirs(OUT, exist_ok=True)


def load_seed(branch, U):
    d = os.path.join(SC, branch, f"U_{U:.1f}")
    R = np.loadtxt(glob.glob(os.path.join(d, "*_R.dat"))[0])
    lam = np.loadtxt(glob.glob(os.path.join(d, "*_lambda.dat"))[0])
    return [R[0], R[1]], [lam[:N], lam[N:]], np.load(os.path.join(d, "z.npy"))


def one_iteration(U, R, lam, ops):
    shift = 1e-9
    Hqp, rho = g.qp_state(R, lam, t)
    Delta = [rho[:N, :N], rho[N:, N:]]
    V = [g.hybridization_V(rho, R, t, shift, i) for i in range(2)]
    lamc = [g.embedding_lambda_c(Delta[i], lam[i], V[i], R[i], shift) for i in range(2)]
    imp = [g.impurity_solve(g.fcidump_round(V[i]), g.fcidump_round(lamc[i]), ops) for i in range(2)]
    Dnew = [np.eye(N) - imp[i][0] for i in range(2)]
    Rnew = [g.apply_S_inv(Dnew[i], imp[i][1], shift) for i in range(2)]
    Lb = np.zeros_like(Hqp); Lb[:N, :N] = lam[0]; Lb[N:, N:] = lam[1]
    E = 2.0 * np.trace(rho @ (Hqp + Lb)) + imp[0][2] + imp[1][2]
    dR = sum(np.linalg.norm(Rnew[i] - R[i]) for i in range(2))
    dD = max(np.abs(Delta[i] - Dnew[i]).max() for i in range(2))
    return E, dR, dD, rho


rows = []
for line in open(os.path.join(SC, "lowest_energy.txt")):
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    U, E_ni, branch = round(float(parts[0]), 1), float(parts[1]), parts[2]
    R, lam, z = load_seed(branch, U)
    ops = g.build_impurity_ops(N, U, U / 2.0)
    model = ni.NotImpDimer(Ng, U, t)
    rho_sol = model.projector(model.unpack(z)[2])[0]

    E1, dR, dD, rho_qp = one_iteration(U, R, lam, ops)
    drho = np.abs(rho_qp - rho_sol).max()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = g.run_gga(Ng, U, t=t, Rg0=R[0], Rg1=R[1], lamg0=lam[0], lamg1=lam[1])
    mm = [float(x) for x in re.findall(r"max\|Delta_qp - Delta_new\| = ([\d.e+-]+)", buf.getvalue())]
    fin = max(mm)   # final constraint mismatch in gga_dimer, worst fragment
    rows.append((U, branch, E_ni, E1, dR, dD, drho, res["converged"], res["iters"] + 1, res["E_var"], fin))
    print(f"U={U:5.1f} {branch:8s} E_direct={E_ni: .8f} | 1 iter: E_seed-E_direct={E1-E_ni: .1e} dR={dR:.1e} "
          f"max|Dqp-Dnew|={dD:.1e} drho={drho:.1e} | run_gga: conv={res['converged']!s:5s} it={res['iters']+1:3d} "
          f"E-E_direct={res['E_var']-E_ni: .1e}", flush=True)

with open(os.path.join(OUT, "gga_dimer_check.txt"), "w") as f:
    f.write("# U branch E_direct E_seed(1 iter)-E_direct dR(1 iter) seed max|Dqp-Dnew| seed drho "
            "converged iters E_gga-E_direct final max|Dqp-Dnew|\n")
    for r in rows:
        f.write(f"{r[0]:5.1f} {r[1]:8s} {r[2]: .8f} {r[3]-r[2]: .3e} {r[4]:.3e} {r[5]:.3e} {r[6]:.3e} {r[7]} {r[8]} "
                f"{r[9]-r[2]: .3e} {r[10]:.3e}\n")

U_ = [r[0] for r in rows]
plt.figure(figsize=(6.4, 4.6))
plt.semilogy(U_, [max(r[5], 1e-16) for r in rows], "o-", ms=3, lw=1, label="seed: max|Delta_qp - Delta_new| (1 iteration)")
plt.semilogy(U_, [max(r[10], 1e-16) for r in rows], "s-", ms=3, lw=1, label="final gga_dimer.py: max|Delta_qp - Delta_new|")
plt.semilogy(U_, [max(abs(r[9] - r[2]), 1e-16) for r in rows], "^-", ms=3, lw=1, label="|E_gga - E_direct|")
plt.xlabel("U/t"); plt.ylabel("value"); plt.legend(fontsize=7)
plt.title("Direct-minimization solutions used as gga_dimer.py seeds")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "gga_dimer_check.png"), dpi=150); plt.close()

arr = np.array([[r[0], r[3]-r[2], r[4], r[5], r[6], float(r[7]), r[8], r[9]-r[2], r[10]] for r in rows])
ok1 = (arr[:, 2] < 1e-4) & (arr[:, 3] < 1e-4)
stay = np.abs(arr[:, 7]) < 1e-6
print(f"\n{len(rows)} U points.  one iteration leaves the seed unchanged (dR<1e-4 and max|Dqp-Dnew|<1e-4): {int(ok1.sum())}")
print(f"run_gga ends at the direct-minimization energy (|dE|<1e-6): {int(stay.sum())};  ends lower: {int((arr[:,7] < -1e-6).sum())};  ends higher: {int((arr[:,7] > 1e-6).sum())}")
print(f"run_gga converged flag True: {int(arr[:,5].sum())}")
print(f"final max|Dqp-Dnew| of run_gga < 1e-4 (constraint satisfied): {int((arr[:,8] < 1e-4).sum())};  > 1e-2: {int((arr[:,8] > 1e-2).sum())}")
