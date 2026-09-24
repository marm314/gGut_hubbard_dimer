"""gga_dimer.py started from the direct-minimization (not-impurity) solutions.

For every U/t = 0.1, 0.2, ..., 10.0 the (R, lambda) reconstructed from the lowest-energy
solution of scan_not_imp/ (scan_not_imp/lowest_energy.txt) is used as the seed of
gga_dimer.run_gga (default settings). For each U it writes scan_mix/U_<U>/ :

  R.in, L.in    the seed (R, lambda) from the direct minimization
  output.out    the full run_gga trace (incl. the Delta_qp / Delta_new comparison) + summary
  gga_Ng2_U*_{R,lambda,lambda_c}.dat   converged (R, lambda, lambda^c) of gga_dimer.py
  eqp.txt       eigenvalues of H_qp(R, lambda) built from the converged gga_dimer.py values

and, for all U together, scan_mix/eqp_vs_U.txt and scan_mix/eqp_vs_U.png (quasiparticle
energies obtained with gga_dimer.py).
"""
import contextlib
import glob
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gga_dimer as g

Ng, t = 2, 1.0
N = Ng + 1
SC, OUT = "scan_not_imp", "scan_mix"
os.makedirs(OUT, exist_ok=True)

rows = []
for line in open(os.path.join(SC, "lowest_energy.txt")):
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    U, branch = round(float(parts[0]), 1), parts[2]
    src = os.path.join(SC, branch, f"U_{U:.1f}")
    R = np.loadtxt(glob.glob(os.path.join(src, "*_R.dat"))[0])
    lam = np.loadtxt(glob.glob(os.path.join(src, "*_lambda.dat"))[0])

    d = os.path.join(OUT, f"U_{U:.1f}")
    os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, "R.in"), R,
               header=f"seed R for U/t={U:.1f}: lowest-energy solution of the direct minimization ({branch})")
    np.savetxt(os.path.join(d, "L.in"), lam,
               header=f"seed lambda for U/t={U:.1f}: lowest-energy solution of the direct minimization ({branch})")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = g.run_gga(Ng, U, t=t, Rg0=R[0], Rg1=R[1], lamg0=lam[:N], lamg1=lam[N:], verbose=True)
    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(buf.getvalue())
        f.write(f"\nNg={Ng}  U/t={U:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
                f"E_var/t={res['E_var']/t:.8f}  docc={res['docc']:.6f}  "
                f"converged={res['converged']}  iters={res['iters'] + 1}  seed from: {branch}\n")
    with contextlib.redirect_stdout(io.StringIO()):
        g.save_results(res, prefix=os.path.join(d, f"gga_Ng{Ng}_U{U:g}"))

    Hqp, rho = g.qp_state([res["R0"], res["R1"]], [res["lam0"], res["lam1"]], t)
    eqp = np.linalg.eigvalsh(Hqp)
    np.savetxt(os.path.join(d, "eqp.txt"), eqp,
               header=f"quasiparticle energies (eigenvalues of H_qp of gga_dimer.py), U/t={U:.4f}")
    rows.append([U] + eqp.tolist())
    print(f"U/t={U:5.1f}  seed:{branch:8s} converged={res['converged']!s:5s} it={res['iters']+1:3d}  "
          f"E_var/t={res['E_var']/t: .8f}  eqp={np.round(eqp, 3)}", flush=True)

rows = np.array(sorted(rows))
np.savetxt(os.path.join(OUT, "eqp_vs_U.txt"), rows,
           header="U/t eqp0..eqp5 (eigenvalues of H_qp of gga_dimer.py started from the direct-minimization solutions)")
plt.figure(figsize=(6.4, 4.8))
for i in range(1, rows.shape[1]):
    plt.plot(rows[:, 0], rows[:, i], "-", lw=1.2)
plt.axhline(0.0, color="gray", lw=0.5, ls=":")
plt.xlabel("U/t")
plt.ylabel("quasiparticle energy / t")
plt.title(f"Quasiparticle energies of gga_dimer.py (Ng={Ng}),\nseeded with the direct-minimization solutions")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eqp_vs_U.png"), dpi=150)
print(f"\nwrote {OUT}/U_*/ , {OUT}/eqp_vs_U.txt and {OUT}/eqp_vs_U.png")
