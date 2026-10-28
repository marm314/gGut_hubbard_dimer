"""Poles/weights of ALL occupied qp levels for the cached minimizer solutions (ghost_solutions.npz; B_c=3, 5; t_c=0.1, 0.05; both V),
no new minimization. Saves cederbaum_data/ghost_solution_poles.npz (keys V{V}_tc{tc}_B{Bc}_poles / _w / _E / _kkt / _comm) and prints the comparison
with the data behind Fig. 2 / Tab. poles (cederbaum_spectra_V*.npz gg_*, cederbaum_bc5_V*.npz). python3 cache_poles.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import gga_cederbaum as g
import not_imp_gga_cederbaum as N
import plot_cederbaum_energies as pe
import plot_cederbaum_spectra as ps
from bc5_cederbaum import embed

Z = np.load("cederbaum_data/ghost_solutions.npz"); S = ps.load_all(); B5 = pe.load_bc5(); out = {}
for TC in (0.1, 0.05):
    for V in pe.VS:
        m3 = g.Model(1.0, 10.0, V, TC, 3, 1e-9, "cv"); P3 = N.Problem(m3); P5, _ = embed(Z[f"V{V}_tc{TC}_z3"], V, TC)
        for Bc, P, z in ((3, P3, Z[f"V{V}_tc{TC}_z3"]), (5, P5, Z[f"V{V}_tc{TC}_z5"])):
            o = N.recover(P, z); E = P.energy_and_grad(z)[0]; k = f"V{V}_tc{TC}_B{Bc}_"
            out[k + "poles"] = o["poles"]; out[k + "w"] = o["weights"]; out[k + "E"] = E; out[k + "kkt"] = o["kkt"]; out[k + "comm"] = o["comm"]
            ref_p, ref_w = (S[f"V{V}_tc{TC}_gg_poles"], S[f"V{V}_tc{TC}_gg_w"]) if Bc == 3 else (B5[f"V{V}_tc{TC}_poles"], B5[f"V{V}_tc{TC}_w"])
            i, j = np.argmax(o["weights"]), np.argmax(ref_w)
            same = len(ref_p) == len(o["poles"]) and np.allclose(ref_p, o["poles"], atol=2e-3) and np.allclose(ref_w[j], o["weights"][i], atol=2e-4)
            print(f"t_c={TC} V={V:.3f} B_c={Bc}: cache main pole {o['poles'][i]:.4f} w {o['weights'][i]:.4f}, levels {np.round(o['poles'],3)} w {np.array2string(o['weights'],precision=1)} | "
                  f"data behind Fig.2/table: main {ref_p[j]:.4f} w {ref_w[j]:.4f}, levels {np.round(ref_p,3)} -> {'SAME' if same else 'DIFFERENT SOLUTION'}", flush=True)
np.savez("cederbaum_data/ghost_solution_poles.npz", **out)
