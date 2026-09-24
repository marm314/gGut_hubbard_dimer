"""U-continuation scan (Ng=2, "forward" branch) from U/t=0.1 up to U/t=10.0
in steps of +0.1.

The "forward" branch (see examples/U2_forward/, doc/code.tex "Are the converged
points solutions?") is only validated at U/t=2 (examples/U2_forward/R.in, L.in).
To reach U/t=0.1 on THIS branch (rather than accidentally seeding the
"backward" branch, see scan_U_backward.py), this script first PRIMES silently: it
continues examples/U2_forward's seed down from U=2 to U=0.1 in steps of -0.1,
without saving anything, using gga_dimer.run_gga (the actual solver,
unmodified) at each step. Only then does the requested ascending scan,
U=0.1 -> 10.0 step +0.1, run and save its output.

For every U/t = U_k of the ascending scan, this writes to scan/forward/U_<U_k>/:
  R.in, L.in    the (R, lambda) guess actually fed into run_gga for this U
                (i.e. the previous point's converged values)
  output.out    the full run_gga verbose trace + a one-line summary
  gga_..._R.dat, _lambda.dat, _lambda_c.dat   the converged (R, lambda, lambda^c),
                via gga_dimer.save_results
  eqp.txt       the 2*Neff eigenvalues of H_qp(R, lambda) at the CONVERGED point
                (the quasiparticle energies), via gga_dimer.qp_state

After the scan, scan/forward/eqp_vs_U.txt collects U and all eqp branches, and
scan/forward/eqp_vs_U.png plots the quasiparticle spectrum vs U.
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
SCAN_DIR = os.path.join("scan", "forward")

os.makedirs(SCAN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Priming: continue the forward-branch seed (validated at U=2) DOWN to U=0.1,
# silently (not saved), so the ascending scan below starts from the correct
# branch instead of an unvalidated guess at U=0.1.
# ---------------------------------------------------------------------------
R0, R1 = g.read_R_guess(os.path.join("examples", "U2_forward", "R.in"))
lam0, lam1 = g.read_lambda_guess(os.path.join("examples", "U2_forward", "L.in"))

Us_prime = [round(u, 1) for u in np.arange(2.0, 0.09, -0.1)]
for U in Us_prime:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = g.run_gga(Ng, U, t=t, Rg0=R0, Rg1=R1, lamg0=lam0, lamg1=lam1, verbose=False)
    if not res["converged"]:
        print(f"[prime] WARNING: not converged at U/t={U}")
    R0, R1, lam0, lam1 = res["R0"], res["R1"], res["lam0"], res["lam1"]
print(f"[prime] reached U/t=0.1 on the forward branch: E_var/t={res['E_var']/t:.6f}  "
      f"Z0={res['Z0']:.6f}  Z1={res['Z1']:.6f}")

# ---------------------------------------------------------------------------
# Requested scan: ascending, U/t = 0.1, 0.2, ..., 10.0
# ---------------------------------------------------------------------------
Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]

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
rows = rows[np.argsort(rows[:, 0])]
header = "U/t " + " ".join(f"eqp{i}" for i in range(rows.shape[1] - 1))
np.savetxt(os.path.join(SCAN_DIR, "eqp_vs_U.txt"), rows, header=header)

plt.figure(figsize=(6, 4.5))
for i in range(1, rows.shape[1]):
    plt.plot(rows[:, 0], rows[:, i], "-", lw=1.2)
plt.xlabel("U/t")
plt.ylabel("quasiparticle energy / t")
plt.title(f"Quasiparticle spectrum of H_qp vs U (Ng={Ng}, forward branch)")
plt.axhline(0.0, color="gray", lw=0.5, ls=":")
plt.tight_layout()
png_path = os.path.join(SCAN_DIR, "eqp_vs_U.png")
plt.savefig(png_path, dpi=150)

print(f"\nwrote {os.path.join(SCAN_DIR, 'eqp_vs_U.txt')} and {png_path}")
