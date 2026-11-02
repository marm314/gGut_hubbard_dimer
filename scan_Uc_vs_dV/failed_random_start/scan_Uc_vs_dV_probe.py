"""Probe (Ng=2): forward U-continuation at one dV, coarse U grid, spectral gap vs U. usage (repo root): python3 scripts/scan_Uc_vs_dV_probe.py DV UMAX DU"""
import sys, os, numpy as np
sys.path.insert(0, "scripts")
import scan_ng2_not_imp_dV2_U as sc, gga_dimer as g
dV, umax, du = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
sc.deltaV = dV; sc.OUT = f"scan_Uc_vs_dV/probe_dV{dV:.1f}"
Us = np.round(np.arange(du, umax + 1e-9, du), 4)
def gap(R, lam, t=1.0):
    Neff = len(R[0]); H = np.zeros((2*Neff, 2*Neff)); H[:Neff,:Neff] = -lam[0]; H[Neff:,Neff:] = -lam[1]
    H[:Neff,Neff:] = -t*np.outer(R[0], R[1]); H[Neff:,:Neff] = H[:Neff,Neff:].T
    eps, phi = np.linalg.eigh(H); Rm = np.zeros((2*Neff, 2)); Rm[:Neff,0] = R[0]; Rm[Neff:,1] = R[1]
    w = 2*((Rm.T @ phi)**2).sum(axis=0)
    pos = [e for e, ww in zip(eps, w) if e > 1e-9 and ww >= 1e-3]; neg = [e for e, ww in zip(eps, w) if e < -1e-9 and ww >= 1e-3]
    return (min(pos) if pos else np.nan) - (max(neg) if neg else np.nan)
model, best = sc.first_point(Us[0], 0); zp = best[1]
for U in Us:
    if U != Us[0]: model, best = sc.continued_point(U, zp, seed=int(U*10)); zp = best[1]
    a = model.analyse(best[1])
    print(f"dV={dV} U={U:6.2f} E={a['E_var']:.6f} Z0={a['Z0']:.3f} Z1={a['Z1']:.3f} sgap={gap([a['R0'],a['R1']],[a['lam0'],a['lam1']]):.4f} conv={sc.ok(best[3],best[4])}", flush=True)
