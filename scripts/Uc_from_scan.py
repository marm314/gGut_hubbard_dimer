"""U_c(dV) from the saved points of scripts/scan_Uc_vs_dV.py (Ng=2): for each dV and U the spectral gap (lowest positive minus highest negative qp level with weight>=1e-3)
of the saved z.npy; U_c = linear extrapolation to zero of the last two grid points with gap>0.01 before the gap disappears (no level with weight) or jumps up by >0.5.
Writes scan_Uc_vs_dV/Uc_vs_dV.txt and gap_vs_U_dV<dV>.txt per dV. Works on whatever is on disk. usage (repo root): python3 scripts/Uc_from_scan.py"""
import glob, os, re, sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
t = 1.0
def gap_E(dV, U):
    d = f"scan_Uc_vs_dV/dV_{dV:.1f}/U_{U:.1f}"; z = np.load(d + "/z.npy"); m = ni.NotImpDimer(2, U, t, deltaV=dV); a = m.analyse(z)
    R, lam = [a["R0"], a["R1"]], [a["lam0"], a["lam1"]]; N = len(R[0])
    H = np.zeros((2*N, 2*N)); H[:N,:N] = -lam[0]; H[N:,N:] = -lam[1]; H[:N,N:] = -t*np.outer(R[0], R[1]); H[N:,:N] = H[:N,N:].T
    eps, phi = np.linalg.eigh(H); Rm = np.zeros((2*N, 2)); Rm[:N,0] = R[0]; Rm[N:,1] = R[1]; w = 2*((Rm.T @ phi)**2).sum(axis=0)
    pos = [e for e, ww in zip(eps, w) if e > 1e-9 and ww >= 1e-3]; neg = [e for e, ww in zip(eps, w) if e < -1e-9 and ww >= 1e-3]
    return (min(pos) if pos else np.nan) - (max(neg) if neg else np.nan), a["E_var"], 0.5 * (a["Z0"] + a["Z1"])
rows = []
for d in sorted(glob.glob("scan_Uc_vs_dV/dV_*"), key=lambda p: float(p.split("_")[-1])):
    dV = float(d.split("_")[-1]); Us = sorted(float(p.split("_")[-1]) for p in glob.glob(d + "/U_*"))
    gf = f"scan_Uc_vs_dV/gap_vs_U_dV{dV:.1f}.txt"
    if os.path.exists(gf) and len(np.loadtxt(gf)) == len(Us): r = np.loadtxt(gf)      # cached (delete the file to recompute)
    else: r = np.array([(U,) + gap_E(dV, U) for U in Us]); np.savetxt(gf, r, header="U/t  spectral_gap  E_var/t  Z")
    U, g = r[:, 0], r[:, 1]
    end = next((i for i in range(1, len(g)) if np.isnan(g[i]) or g[i] > g[i-1] + 0.5), None)
    if end is None: rows.append((dV, np.nan)); print(f"dV={dV:.1f}: no closing up to U={U[-1]}"); continue
    ok = [i for i in range(end) if g[i] > 0.01][-2:]
    uc = U[ok[1]] - g[ok[1]] * (U[ok[1]] - U[ok[0]]) / (g[ok[1]] - g[ok[0]])
    rows.append((dV, uc)); print(f"dV={dV:.1f}: U_c/t={uc:.3f} (gap closes after U={U[end-1]:.1f}, {len(U)} points)", flush=True)
np.savetxt("scan_Uc_vs_dV/Uc_vs_dV.txt", np.array(rows), header="dV/t  Uc/t (nan: no closing)")
