"""Ionic vs covalent weight of the Hubbard dimer ground state, exact (ED) and gGA, as U/t grows.
ionic weight P_ion = <n_0up n_0dn> + <n_1up n_1dn> (probability that one site is doubly occupied, i.e. the H- H+ configurations),
covalent weight = 1 - P_ion (one electron on each site), n_site = electrons on a site.
gGA: docc_i and n_phys_i come from the impurity ground state of each fragment (gga_dimer.impurity_solve) built from the converged (R, lambda)
exactly as in one iteration of gga_dimer.run_gga (qp_state -> Delta -> V, lambda^c)."""
import sys
import numpy as np
import gga_dimer as gg
import exact_dimer_deltaV as ed
from make_qp_orbital_videos import load_R_lambda

t = 1.0


def ed_weights(U, dV_ed):
    H, c, n0, n1 = ed.build_hamiltonian(t, U, dV_ed)
    N = n0 + n1
    nd = np.real(np.diag(N))
    sec = np.flatnonzero(np.abs(nd - 2) < 1e-9)
    w, v = np.linalg.eigh(H[np.ix_(sec, sec)])
    gs = np.zeros(H.shape[0]); gs[sec] = v[:, 0]
    n0u = c[0].conj().T @ c[0]; n0d = c[1].conj().T @ c[1]; n1u = c[2].conj().T @ c[2]; n1d = c[3].conj().T @ c[3]
    d0 = gs @ (n0u @ n0d @ gs); d1 = gs @ (n1u @ n1d @ gs)
    return d0 + d1, gs @ (n0 @ gs)


def gga_weights(d, U, dV):
    R, lam = load_R_lambda(d); Neff = len(R[0])
    ops = [gg.build_impurity_ops(Neff, U, U / 2 + dV), gg.build_impurity_ops(Neff, U, U / 2 - dV)]
    Hqp, rho = gg.qp_state(R, lam, t)
    blocks = [rho[:Neff, :Neff], rho[Neff:, Neff:]]
    V = [gg.hybridization_V(rho, R, t, 1e-9, i) for i in range(2)]
    lamc = [gg.embedding_lambda_c(blocks[i], lam[i], V[i], R[i], 1e-9) for i in range(2)]
    imp = [gg.impurity_solve(V[i], lamc[i], ops[i]) for i in range(2)]
    Lblk = np.zeros_like(Hqp); Lblk[:Neff, :Neff] = lam[0]; Lblk[Neff:, Neff:] = lam[1]
    E_var = 2.0 * np.trace(rho @ (Hqp + Lblk)) + imp[0][2] + imp[1][2]       # as gga_dimer.run_gga
    gga_weights.E_var = E_var
    return imp[0][3] + imp[1][3], imp[0][4], imp[1][4], R


if __name__ == "__main__":
    which = sys.argv[1]; ng = int(sys.argv[2])
    br2 = {}
    if which == "dV0":
        for line in open("scan_ng2_not_imp/lowest_energy.txt"):
            if not line.startswith("#") and line.strip():
                f = line.split(); br2[round(float(f[0]), 1)] = f[2]
        dirf = (lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}") if ng == 2 else (lambda U: f"scan_ng4_not_imp/U_{U:.1f}")
        dV = 0.0
    else:
        dirf = lambda U: f"scan_ng{ng}_dV2_confirmed/U_{U:.1f}"; dV = 2.0
    print("   U   |  ED: P_ion  n_site0 |  gGA: P_ion  n_site0  n_site1   (header docc*2)")
    for U in (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0):
        pe, ne = ed_weights(U, 2 * dV)
        pg, n0, n1, R = gga_weights(dirf(U), U, dV)
        import re
        hdr = open(dirf(U) + "/output.out").read(); m = re.search(r"docc=([\d.eE+-]+)", hdr)
        print(f" {U:5.1f} |  {pe:.4f}  {ne:.4f}   |  {pg:.4f}  {n0:.4f}  {n1:.4f}   ({2*float(m.group(1)):.4f})" if m else f" {U:5.1f} |  {pe:.4f}  {ne:.4f}   |  {pg:.4f}  {n0:.4f}  {n1:.4f}", flush=True)
