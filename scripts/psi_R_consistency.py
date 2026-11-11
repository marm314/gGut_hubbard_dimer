"""Self-consistency of a hand-modified R (mode B): R'_i put into H_qp, V, lambda^c vs R_Phi,i recomputed from the re-solved embedding state (Rchk of build_psi).
Usage: python psi_R_consistency.py <dV0|dV2> <Ng> <U> <part: none|rot>. Output ../psi_scan/Rcons_ng<Ng>_<tag>_U<U>_<part>.txt:
s, P_ion, |R'_0|, |R_Phi,0|, |R_Phi,0 - R'_0|, then R'_0 (Ng+1 values) and R_Phi,0 (Ng+1 values)."""
import sys
import numpy as np
import gutzwiller_psi as G
tag, ng, U, part = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d, dV = G.solution_dir(tag, ng, U)
out = f"../psi_scan/Rcons_ng{ng}_{tag}_U{U:.1f}_{part}.txt"
open(out, "w").write("# s Pion |Rin| |Rphi| |Rphi-Rin| Rin(Neff) Rphi(Neff)   [site 0]\n")
for s in np.round(np.arange(0.7, 1.301, 0.05), 2):
    try:
        r = G.build_psi(d, U, dV, Ng=ng, verbose=False, Rscale=(s, s), mode="B", part=None if part == "none" else part)
        a = G.analyse(r["Psi"], U, dV)
    except Exception as e:
        print(f"s={s}: FAILED {e!r}", flush=True); continue
    Rin = np.array(r["R"][0]); Rp = np.array(r["info"][0]["Rchk"])
    row = [s, a["pion"], np.linalg.norm(Rin), np.linalg.norm(Rp), np.linalg.norm(Rp - Rin)] + list(Rin) + list(Rp)
    open(out, "a").write(" ".join(f"{x:.8g}" for x in row) + "\n")
    print(f"s={s:.2f} P_ion={a['pion']:.4f} |R_in|={row[2]:.4f} |R_Phi|={row[3]:.4f} |R_Phi-R_in|={row[4]:.4f}", flush=True)
