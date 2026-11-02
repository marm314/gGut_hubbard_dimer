"""U_c at dV=10 from the saved points of scripts/scan_Uc_dV10.py (same gap definition and extrapolation as Uc_from_scan.py). usage (repo root): python3 scripts/Uc_dV10.py"""
import glob, sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
t = 1.0; dV = 10.0
def gap_E(U):
    z = np.load(f"scan_Uc_dV10/dV_10.0/U_{U:.1f}/z.npy"); a = ni.NotImpDimer(2, U, t, deltaV=dV).analyse(z)
    R, lam = [a["R0"], a["R1"]], [a["lam0"], a["lam1"]]; N = len(R[0])
    H = np.zeros((2*N, 2*N)); H[:N,:N] = -lam[0]; H[N:,N:] = -lam[1]; H[:N,N:] = -t*np.outer(R[0], R[1]); H[N:,:N] = H[:N,N:].T
    eps, phi = np.linalg.eigh(H); Rm = np.zeros((2*N, 2)); Rm[:N,0] = R[0]; Rm[N:,1] = R[1]; w = 2*((Rm.T @ phi)**2).sum(axis=0)
    pos = [e for e, ww in zip(eps, w) if e > 1e-9 and ww >= 1e-3]; neg = [e for e, ww in zip(eps, w) if e < -1e-9 and ww >= 1e-3]
    return (min(pos) if pos else np.nan) - (max(neg) if neg else np.nan), a["E_var"], 0.5 * (a["Z0"] + a["Z1"])
Us = sorted(float(p.split("_")[-1]) for p in glob.glob("scan_Uc_dV10/dV_10.0/U_*"))
r = np.array([(U,) + gap_E(U) for U in Us]); np.savetxt("scan_Uc_dV10/gap_vs_U_dV10.txt", r, header="U/t spectral_gap E_var/t Z")
U, g = r[:, 0], r[:, 1]; end = next((i for i in range(1, len(g)) if np.isnan(g[i]) or g[i] > g[i-1] + 0.5), None)
if end is None: print("no closing up to U =", U[-1]); sys.exit()
ok = [i for i in range(end) if g[i] > 0.01][-2:]; uc = U[ok[1]] - g[ok[1]] * (U[ok[1]] - U[ok[0]]) / (g[ok[1]] - g[ok[0]])
np.savetxt("scan_Uc_dV10/Uc_dV10.txt", [[dV, uc]], header="dV/t Uc/t")
print(f"dV=10: U_c/t={uc:.3f} (gap closes after U={U[end-1]:.1f}; last gaps {g[ok[0]]:.4f}, {g[ok[1]]:.4f}; next U={U[end]:.1f} gap={g[end]:.4f})")
