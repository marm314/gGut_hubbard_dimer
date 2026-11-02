"""P_ion from the Phi-matrix quadratic form of the direct-minimization solutions (saved z.npy):
P_ion = phi_0^T N_dd phi_0 + phi_1^T N_dd phi_1, no impurity solve. Compared with the impurity-ground-state value
(merged_Ng/ionic_spectral_link_data.npz) and ED. Run from the repo root. Writes merged_Ng/pion_from_phi.npz and prints a table."""
import sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
import ionic_covalent as ic

Us = np.round(np.arange(0.1, 10.01, 0.1), 1)
ref = np.load("merged_Ng/ionic_spectral_link_data.npz")
out = {"U": Us}
for dV, root in ((0, "scan_ng2_not_imp"), (2, "scan_ng2_not_imp_dV2")):
    br = {}
    for line in open(f"{root}/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            f = line.split(); br[round(float(f[0]), 1)] = f[2]
    pphi, site = [], []
    for U in Us:
        m = ni.NotImpDimer(2, float(U), deltaV=float(dV))
        z = np.load(f"{root}/{br[U]}/U_{U:.1f}/z.npy")
        x0, x1, _ = m.unpack(z)
        d = [(x @ m.Ndd @ x) / (x @ x) for x in (x0, x1)]
        pphi.append(d[0] + d[1]); site.append(d)
        print(f"dV={dV} U={U:.1f} {br[U]:8s} Pion(phi)={pphi[-1]:.6f} imp={ref[f'pion_gga_dV{dV}'][list(Us).index(U)]:.6f} ED={ref[f'pion_ed_dV{dV}'][list(Us).index(U)]:.6f}", flush=True)
    out[f"pion_phi_dV{dV}"] = np.array(pphi); out[f"site_phi_dV{dV}"] = np.array(site)
np.savez("merged_Ng/pion_from_phi.npz", **out)
