"""Ng=0 orbitals versus the Ng=2 HOMO and LUMO (dV=0 and 2): for the tracked Ng=2 HOMO/LUMO, the weight n_s on the two states, the overlap of their state part with the
Ng=0 occupied/virtual orbital (analytic, scripts/ng0_analytic.py), the level energies and the A(omega) weights of both. Writes merged_Ng/orbital_comparison_ng0_ng2.txt. Run from the repo root."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import make_qp_orbital_videos as mo
from ng0_analytic import ng0_state
br2 = {}
for line in open("scan_ng2_not_imp/lowest_energy.txt"):
    if not line.startswith("#") and line.strip(): f = line.split(); br2[round(float(f[0]), 1)] = f[2]
rows = []
for dV, get in ((0.0, lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}"), (2.0, lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}")):
    for f in mo.load_frames(get):
        Neff = f["Neff"]; occ = f["occf"] > 0.5; ih = int(np.argmax(np.where(occ, f["eps"], -np.inf)))
        a = ng0_state(f["U"], dV); D0, D1 = (1 + a["m"]) / 2, (1 - a["m"]) / 2; o = np.array([np.sqrt(D0), np.sqrt(D1)])
        c = f["ph"][[0, Neff], ih]; ns = float(c @ c); ov = abs(c @ o) / np.sqrt(ns) if ns > 1e-9 else float("nan")
        rows.append((dV, f["U"], ns, ov, abs(f["eps"][ih]), a["eps"], f["w"][ih], a["w"]))
np.savetxt("merged_Ng/orbital_comparison_ng0_ng2.txt", np.array(rows), header="dV U/t n_s(Ng=2 HOMO on the two states) overlap(state part, Ng=0 occ) |eps|(Ng=2) |eps|(Ng=0) A-weight(Ng=2) A-weight(Ng=0)")
for dV in (0.0, 2.0):
    print("dV =", dV)
    for r in rows:
        if r[0] == dV and r[1] in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0): print("  U=%.0f  n_s=%.2f overlap=%.3f eps2=%.2f eps0=%.2f w2=%.2f w0=%.2f" % r[1:])
