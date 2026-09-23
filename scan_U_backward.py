"""U-continuation scan (Ng=2) from U/t=10 down to U/t=0.1 in steps of -0.1.

For each U, gga_dimer.run_gga (the actual solver, unmodified) is called with
the PREVIOUS U's converged (R, lambda) as the seed -- a manual continuation
in U, since a generic cold start is not reliable everywhere (see doc/code.tex).
The very first point (U=10) uses the seed recovered from the old
examples/Ng2_U10/R.in, L.in (git commit 2c46ccc~1).

For every U/t = U_k, this writes to scan/U_<U_k>/:
  R.in, L.in    the (R, lambda) guess actually fed into run_gga for this U
                (i.e. the previous point's converged values)
  output.out    the full run_gga verbose trace + a one-line summary
  gga_..._R.dat, _lambda.dat, _lambda_c.dat   the converged (R, lambda, lambda^c),
                via gga_dimer.save_results
  eqp.txt       the 2*Neff eigenvalues of H_qp(R, lambda) at the CONVERGED point
                (the quasiparticle energies), via gga_dimer.qp_state

After the scan, scan/eqp_vs_U.txt collects U and all eqp branches, and
scan/eqp_vs_U.png plots the quasiparticle spectrum vs U.
"""
import contextlib
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gga_dimer as g

Ng, t = 2, 1.0
Neff = Ng + 1
SCAN_DIR = "scan"

os.makedirs(SCAN_DIR, exist_ok=True)

# Seed at U=10 (recovered from git history: examples/Ng2_U10/R.in, L.in
# before the repo was reorganized around the forward/backward U=2 examples).
R0 = np.array([0.696171, 0.696171, 0.150289])
R1 = R0.copy()
lam0 = np.array([[4.70937, 0.0, 0.510158],
                  [0.0, -4.70937, -0.510158],
                  [0.510158, -0.510158, 0.0]])
lam1 = lam0.copy()

Us = [round(u, 1) for u in np.arange(10.0, 0.09, -0.1)]  # 10.0, 9.9, ..., 0.1

rows = []
for U in Us:
    d = os.path.join(SCAN_DIR, f"U_{U:.1f}")
    os.makedirs(d, exist_ok=True)

    np.savetxt(os.path.join(d, "R.in"), np.vstack([R0, R1]),
               header=f"input R guess for U/t={U:.1f} (previous point's converged R)")
    np.savetxt(os.path.join(d, "L.in"), np.vstack([lam0, lam1]),
               header=f"input lambda guess for U/t={U:.1f} (previous point's converged lambda)")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = g.run_gga(Ng, U, t=t, Rg0=R0, Rg1=R1, lamg0=lam0, lamg1=lam1, verbose=True)

    with open(os.path.join(d, "output.out"), "w") as f:
        f.write(buf.getvalue())
        f.write(f"\nNg={Ng}  U/t={U:.4f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}  "
                f"E_var/t={res['E_var']/t:.6f}  docc={res['docc']:.6f}  "
                f"converged={res['converged']}  iters={res['iters']}\n")

    g.save_results(res, prefix=os.path.join(d, f"gga_Ng{Ng}_U{U:g}"))

    R0, R1 = res["R0"], res["R1"]
    lam0, lam1 = res["lam0"], res["lam1"]

    Hqp, rho = g.qp_state([R0, R1], [lam0, lam1], t)
    eqp = np.linalg.eigvalsh(Hqp)
    np.savetxt(os.path.join(d, "eqp.txt"), eqp,
               header=f"quasiparticle energies (eigenvalues of H_qp), U/t={U:.4f}")

    rows.append([U] + eqp.tolist())
    print(f"U/t={U:5.1f}  converged={res['converged']!s:5s}  iters={res['iters']:3d}  "
          f"E_var/t={res['E_var']/t: .6f}  Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}", flush=True)

rows = np.array(rows)
rows = rows[np.argsort(rows[:, 0])]  # ascending U for the summary file / plot
header = "U/t " + " ".join(f"eqp{i}" for i in range(rows.shape[1] - 1))
np.savetxt(os.path.join(SCAN_DIR, "eqp_vs_U.txt"), rows, header=header)

plt.figure(figsize=(6, 4.5))
for i in range(1, rows.shape[1]):
    plt.plot(rows[:, 0], rows[:, i], "-", lw=1.2)
plt.xlabel("U/t")
plt.ylabel("quasiparticle energy / t")
plt.title(f"Quasiparticle spectrum of H_qp vs U (Ng={Ng})")
plt.axhline(0.0, color="gray", lw=0.5, ls=":")
plt.tight_layout()
png_path = os.path.join(SCAN_DIR, "eqp_vs_U.png")
plt.savefig(png_path, dpi=150)

print(f"\nwrote {os.path.join(SCAN_DIR, 'eqp_vs_U.txt')} and {png_path}")
