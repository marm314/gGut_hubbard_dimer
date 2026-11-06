"""Norm of the Gutzwiller wave function Psi = P_0 P_1 |Psi0>, dV=0: <Psi|Psi> = <Psi0| G_0 G_1 |Psi0>, G_i = P_i^dag P_i = P0_i^-1/2 Phit_i^T Phit_i P0_i^-1/2.
Prints <G_0>, <G_1> (single-fragment norms, =1 by the gGA constraints), <G_0 G_1> (= <Psi|Psi>) and the connected part <G_0 G_1> - <G_0><G_1>;
for Ng=0 also the closed form 1 + (U/Uc)^2, Uc = 8. Usage: python psi_norm_check.py <Ng> U [U ...]   (output: ../psi_scan/psi_norm_check.txt)"""
import sys
import numpy as np
import gutzwiller_psi as G

ng = int(sys.argv[1]); out = open("../psi_scan/psi_norm_check.txt", "a")
for U in (float(x) for x in sys.argv[2:]):
    d, dV = G.solution_dir("dV0", ng, U)
    r = G.build_psi(d, U, dV, Ng=ng, verbose=False)
    L0, L1, M = r["lams"][0], r["lams"][1], r["M"]
    G0, G1 = L0.T @ L0, L1.T @ L1                       # operators on the qp Fock space of each fragment
    g0 = np.sum(M * (G0 @ M)); g1 = np.sum(M * (M @ G1.T)); g01 = np.sum(M * (G0 @ M @ G1.T))
    nrm = float(r["Psi"] @ r["Psi"])
    line = f"Ng={ng} U/t={U}: <G0>={g0:.10f} <G1>={g1:.10f} <G0G1>={g01:.10f} <Psi|Psi>={nrm:.10f} connected={g01-g0*g1:.10f}"
    if ng == 0:
        line += f"  1+(U/8)^2={1+(U/8)**2:.10f}"
    print(line); out.write(line + "\n")
