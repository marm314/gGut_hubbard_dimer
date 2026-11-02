"""Ng=0 analysis from the saved scans of scripts/scan_ng0.py (lowest-energy branch per U): per dV a table
U, E_var, Z0, Z1, docc (per site, mean), P_ion (=2 docc), qp levels (eigenvalues of H_qp), level weights (2*sum |R-component|^2, as in make_qp_orbital_videos), and U_c
(= where the qp weights, i.e. Z, vanish: linear extrapolation to Z=0 of the last two grid points with Z>1e-4). Writes scan_ng0/table_dV<dV>.txt and scan_ng0/Uc_ng0_vs_dV.txt.
Also the closed forms of gGut_notes.pdf (Sec. III D) at dV=0 are checked against the numerics. usage (repo root): python3 scripts/ng0_analysis.py"""
import glob, sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
t = 1.0
def row(dV, U, branch):
    z = np.load(f"scan_ng0/dV_{dV:.1f}/{branch}/U_{U:.1f}/z.npy"); m = ni.NotImpDimer(0, U, t, deltaV=dV); a = m.analyse(z)
    R, lam = [a["R0"], a["R1"]], [a["lam0"], a["lam1"]]; N = 1
    H = np.zeros((2, 2)); H[0, 0] = -lam[0][0, 0]; H[1, 1] = -lam[1][0, 0]; H[0, 1] = H[1, 0] = -t * R[0][0] * R[1][0]
    eps, phi = np.linalg.eigh(H); w = 2 * ((R[0][0] * phi[0]) ** 2 + (R[1][0] * phi[1]) ** 2)
    return [U, a["E_var"], a["Z0"], a["Z1"], a["docc"], 2 * a["docc"], eps[0], eps[1], w[0], w[1]]
Uc = []
for d in sorted(glob.glob("scan_ng0/dV_*"), key=lambda p: float(p.split("_")[-1])):
    dV = float(d.split("_")[-1]); lo = [l.split() for l in open(d + "/lowest_energy.txt") if not l.startswith("#") and l.strip()]
    r = np.array([row(dV, float(f[0]), f[2]) for f in lo]); np.savetxt(f"scan_ng0/table_dV{dV:.1f}.txt", r, header="U/t E_var/t Z0 Z1 docc Pion eps0 eps1 w0 w1")
    Z = 0.5 * (r[:, 2] + r[:, 3]); k = np.flatnonzero(Z > 1e-4)[-1]; U1, U2, Z1, Z2 = r[k - 1, 0], r[k, 0], Z[k - 1], Z[k]
    uc = U2 - Z2 * (U2 - U1) / (Z2 - Z1); Uc.append((dV, uc)); print(f"dV={dV:.1f}: U_c/t={uc:.3f} (Z={Z2:.4f} at U={U2:.1f}, {len(r)} points)", flush=True)
np.savetxt("scan_ng0/Uc_ng0_vs_dV.txt", np.array(Uc), header="dV/t Uc/t (Z -> 0)")
r = np.loadtxt("scan_ng0/table_dV0.0.txt"); U = r[:, 0]; ok = U <= 8
Za = np.where(ok, 1 - (U / 8) ** 2, 0.0); Ea = np.where(ok, -2 - U / 2 - U ** 2 / 32, -U); da = np.where(ok, 0.25 - U / 32, 0.0)
print("closed form vs numerics at dV=0 (max abs diff): Z %.1e  E %.1e  docc %.1e" % (abs(Za - r[:, 2]).max(), abs(Ea - r[:, 1]).max(), abs(da - r[:, 4]).max()))
