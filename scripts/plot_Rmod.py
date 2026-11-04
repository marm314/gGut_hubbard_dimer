"""P_ion of Psi vs the weight x = |R_HL|^2 of R_i on the HOMO/LUMO-like orbitals, norm |R_i| kept (Ng=2, mode A, part=rot; from psi_R_modify.py) against Ng=0 (x = R^2 = Z, mode B)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gutzwiller_psi as G

def share(dv, U):
    d, _ = G.solution_dir(dv, 2, U); R, lam = G.load_R_lambda(d); N = 3
    W = np.linalg.eigh(np.block([[-lam[0], -np.outer(R[0], R[1])], [-np.outer(R[1], R[0]), -lam[1]]]))[1]
    B = np.column_stack([W[:N, k] for k in (2, 3)]); Q, sv, _ = np.linalg.svd(B, full_matrices=False); Q = Q[:, sv > 1e-8]
    return float(np.linalg.norm(R[0] @ Q @ Q.T) ** 2), float(R[0] @ R[0])

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
    cols = {2.0: "#1b9e77", 4.0: "#d95f02", 6.0: "#7570b3"}
    for ax, dv in zip(axs, ("dV0", "dV2")):
        for U, c in cols.items():
            sh, n2 = share(dv, U)
            a = [l.split() for l in open(f"../psi_scan/Rmod_ng2_{dv}_U{U:.1f}_both_rot.txt") if l[0] == "A"]
            s = np.array([float(r[1]) for r in a]); p = np.array([float(r[3]) for r in a]); o = np.argsort(s)
            ax.plot((s * s * sh)[o], p[o], "o-", color=c, label=f"Ng=2, U/t={U:g}")
            ax.plot([sh], [p[np.argmin(abs(s - 1))]], "k*", ms=10)
        if dv == "dV0":
            a = [l.split() for l in open("../psi_scan/Rmod_ng0_dV0_U4.0_both.txt") if l[0] == "B"]
            s = np.array([float(r[1]) for r in a]); p = np.array([float(r[3]) for r in a])
            ax.plot(0.75 * s * s, p, "s--", color="gray", label="Ng=0, U/t=4 (x=R$^2$)")
        ax.set_xlabel("weight of R on HOMO/LUMO,  $|R_{HL}|^2$"); ax.set_ylabel("$P_{ion}(\\Psi_G)$"); ax.set_title(f"$\\Delta V$={dv[2:]}"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("../merged_Ng/Rmod_pion_vs_Rweight.png", dpi=130)
