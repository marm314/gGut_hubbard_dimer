"""Gutzwiller wave function Psi (gutzwiller_psi.py) over the whole U/t grid of the gGA scans, for Ng=0 and 2 at dV=0 and 2.
Every point is appended to psi_scan/ng<Ng>_dV<dV>.txt as soon as it is computed (and Psi to the .npz), the progress goes to the log;
failed points are listed at the end. Usage: python psi_scan.py <Ng> <dV0|dV2|dV2n>   (dV2: impurity-solver scan_ng<Ng>_dV2_confirmed; dV2n: non-impurity scan_ng<Ng>_not_imp_dV2)
Columns: U/t, norm <Psi|Psi>, E_Psi, E_FCI, E_gGA, overlap with the exact ground state, P_ion(Psi), P_ion(exact), P_ion(fragments),
cov and ionic (symmetric, antisymmetric) amplitudes of normalized Psi, kinetic energy of Psi / exact / gGA functional,
weight of Phit on the null space of P0, then the same three amplitudes of the exact ground state (global sign: cov > 0)."""
import glob
import os
import sys
import time
import numpy as np
import gutzwiller_psi as G

ng, which = int(sys.argv[1]), sys.argv[2]
out = "../psi_scan"; os.makedirs(out, exist_ok=True)
tag = f"ng{ng}_{which}"
if ng == 0:
    dV = 0.0 if which == "dV0" else 2.0
    Us = sorted(G._branches(f"../scan_ng0/dV_{dV:.1f}/lowest_energy.txt"))
elif which == "dV2n":
    Us = sorted(G._branches(f"../scan_ng{ng}_not_imp_dV2/lowest_energy.txt"))
elif which == "dV0":
    Us = sorted(G._branches("../scan_ng2_not_imp/lowest_energy.txt"))
else:
    Us = sorted(round(float(p.split("U_")[-1]), 1) for p in glob.glob(f"../scan_ng{ng}_dV2_confirmed/U_*"))
hdr = ("# U/t  norm  E_Psi  E_FCI  E_gGA  overlap  Pion_Psi  Pion_exact  Pion_frag  amp_cov  amp_ion_sym  amp_ion_asym  "
       "kin_Psi  kin_exact  kin_gGA  wnull  amp_cov_exact  amp_ion_sym_exact  amp_ion_asym_exact\n")
txt = f"{out}/{tag}.txt"
open(txt, "w").write(hdr)
psis, failed = {}, []
t0 = time.time()
for k, U in enumerate(Us):
    try:
        a = G.report(which, ng, U, verbose=False)
    except Exception as e:                                            # missing / unreadable point
        failed.append((U, repr(e)[:80])); print(f"[{k+1}/{len(Us)}] U/t={U}: FAILED {e!r}", flush=True); continue
    row = [U, a["norm"], a["E"], a["E_ed"], a["E_gga"], a["ovl"], a["pion"], a["pion_ed"], a["pion_frag"], a["amp_cov"], a["amp_ion"],
           a["amp_ionasym"], a["kin"], a["kin_ed"], a["kin_gga"], a["wnull"], a["amp_cov_ed"], a["amp_ion_ed"], a["amp_ionasym_ed"]]
    with open(txt, "a") as f:
        f.write(" ".join(f"{x:.10g}" for x in row) + "\n")
    psis[f"U{U:.1f}"] = a["p"]; np.savez(f"{out}/{tag}_psi.npz", **psis)
    print(f"[{k+1}/{len(Us)}] U/t={U}: norm={a['norm']:.4f} E_Psi-E_FCI={a['E']-a['E_ed']:.2e} overlap={a['ovl']:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)
print(f"done: {len(psis)} converged points, {len(failed)} failed: {failed}", flush=True)
