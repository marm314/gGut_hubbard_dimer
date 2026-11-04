"""Hand-modified R in the Gutzwiller wave function Psi (gutzwiller_psi.build_psi, Rscale / mode): R_i -> s_i R_i at fixed lambda, from a converged gGA point.
mode A: only H_qp sees the modified R (Psi0, M, P0 change; the embedding states Phi_i are those of the solution);
mode B: R modified everywhere (rho, V, lambda^c) and the embedding ground states re-solved (one loop iteration with the modified R).
Usage: python psi_R_modify.py <dV0|dV2|dV2n> <Ng> <U> [which=both|0|1] [part=hl|ghost: scale only the HOMO/LUMO-span or the ghost part of R_i]   (scales 0.7 ... 1.3; which = rescaled site(s))
Every row is appended to ../psi_scan/Rmod_ng<Ng>_<tag>_U<U>_<which>.txt as soon as computed.
Columns: mode, s, norm, P_ion, amp_cov, amp_ion_sym, amp_ion_asym, E_Psi, overlap with the exact ground state, kin, R0chk, R1chk (R from Phit)."""
import os
import sys
import time
import numpy as np
import gutzwiller_psi as G

tag, ng, U = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
which = sys.argv[4] if len(sys.argv) > 4 else "both"
part = sys.argv[5] if len(sys.argv) > 5 else None          # None | hl | ghost
d, dV = G.solution_dir(tag, ng, U)
scales = np.round(np.arange(0.70, 1.301, 0.05), 2)
if ng == 0:                                  # no R / lambda files for Ng=0: R_i = sqrt(Z_i) from output.out, lambda = 0 (dV=0 only)
    import re, shutil
    assert dV == 0.0, "Ng=0 with dV != 0 needs lambda (not saved)"
    h = open(d + "/output.out").read()
    Z = [float(re.search(rf"Z{i}=([\d.eE+-]+)", h).group(1)) for i in range(2)]
    sd = f"/tmp/claude-1000/-home-marm314-Installers-gGut-mau/8d56d85f-07f9-4263-a7ef-3edd904cd99d/scratchpad/ng0_U{U:.1f}"
    os.makedirs(sd, exist_ok=True)
    np.savetxt(sd + "/x_R.dat", np.sqrt(Z).reshape(2, 1)); np.savetxt(sd + "/x_lambda.dat", np.zeros((2, 1)))
    d = sd
    scales = np.round(np.arange(0.1, 1.501, 0.1), 2)
    print(f"Z={Z}", flush=True)
out = f"../psi_scan/Rmod_ng{ng}_{tag}_U{U:.1f}_{which}{"_" + part if part else ""}.txt"
os.makedirs("../psi_scan", exist_ok=True)
open(out, "w").write("# mode s norm Pion amp_cov amp_ion_sym amp_ion_asym E_Psi overlap kin R0chk R1chk\n")
print(f"{tag} Ng={ng} U/t={U} dV={dV}  dir={d}", flush=True)
t0 = time.time()
for mode in "AB":
    for s in scales:
        sc = (s if which in ("both", "0") else 1.0, s if which in ("both", "1") else 1.0)
        try:
            r = G.build_psi(d, U, dV, Ng=ng, verbose=False, Rscale=sc, mode=mode, part=part)
            a = G.analyse(r["Psi"], U, dV)
        except Exception as e:
            print(f"mode {mode} s={s}: FAILED {e!r}", flush=True); continue
        Rc = [i["Rchk"][0] for i in r["info"]]
        row = [a["norm"], a["pion"], a["amp_cov"], a["amp_ion"], a["amp_ionasym"], a["E"], a["ovl"], a["kin"], Rc[0], Rc[1]]
        open(out, "a").write(f"{mode} {s:.2f} " + " ".join(f"{x:.10g}" for x in row) + "\n")
        print(f"mode {mode} s={s:.2f}: norm={a['norm']:.4f} P_ion={a['pion']:.4f} cov={a['amp_cov']:+.4f} ion={a['amp_ion']:+.4f} "
              f"E={a['E']:.5f} ovl={a['ovl']:.5f} ({time.time()-t0:.0f}s)", flush=True)
print(f"exact: P_ion={a['pion_ed']:.4f} cov={a['amp_cov_ed']:+.4f} E={a['E_ed']:.5f}")
