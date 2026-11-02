"""P_ion from the Phi-matrix form at the CONFIRMED dV=2 solutions (scan_ng2_dV2_confirmed): seed NotImpDimer at the confirmed (R,lambda)
(start_from_R_lambda), report phi^T N_dd phi at the seed and after the direct minimization from that seed, with E_var of each,
next to the impurity value of the confirmed point. Run from the repo root. Writes merged_Ng/pion_from_phi_confirmed.npz."""
import sys, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
import make_qp_orbital_videos as q

ref = np.load("merged_Ng/ionic_spectral_link_data.npz"); Us = ref["U"]
rows = []
for k, U in enumerate(Us):
    d = f"scan_ng2_dV2_confirmed/U_{U:.1f}"
    R, lam = q.load_R_lambda(d)
    m = ni.NotImpDimer(2, float(U), deltaV=2.0)
    z0 = m.start_from_R_lambda(R, lam)
    p0 = sum((x @ m.Ndd @ x) / (x @ x) for x in m.unpack(z0)[:2])
    E0 = m.energy(z0, grad=False) if False else m.energy(z0)[0]
    res = m.minimize(z0); z1 = res.x
    p1 = sum((x @ m.Ndd @ x) / (x @ x) for x in m.unpack(z1)[:2]); E1 = m.energy(z1)[0]
    rows.append((U, p0, E0, p1, E1, ref["pion_gga_dV2"][k]))
    print(f"U={U:.1f} seed: Pion={p0:.6f} E={E0:.8f} |c|={np.abs(m.constraints(z0)).max():.1e} | min: Pion={p1:.6f} E={E1:.8f} | imp={ref['pion_gga_dV2'][k]:.6f}", flush=True)
np.savez("merged_Ng/pion_from_phi_confirmed.npz", rows=np.array(rows))
