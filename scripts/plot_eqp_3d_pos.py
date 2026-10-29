"""Fig. 3 of doc/results.tex with only the non-negative quasiparticle energies (A(omega) is symmetric about 0).
Writes eqp_vs_U_3d_pos.png next to each eqp_vs_U_3d.png (the four panels of the figure)."""
import plot_eqp_3d as p
import plot_eqp_3d_dV2 as q

Us = p.Us_ridge
br = {}
for line in open("scan_ng2_not_imp/lowest_energy.txt"):
    if not line.startswith("#") and line.strip():
        f = line.split(); br[round(float(f[0]), 1)] = f[2]
p.ridge_plot(lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}", Us, "Spectral function, lowest-energy solution (Ng=2)",
             "scan_ng2_not_imp/eqp_vs_U_3d_pos.png", omega_min=0.0)
p.ridge_plot(lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", Us, "Ng=2, dV=2",
             "scan_ng2_dV2_confirmed/eqp_vs_U_3d_pos.png", ed_fn=q.ed_Aomega_dV2, omega_min=0.0)
p.ridge_plot(lambda U: f"scan_ng4_not_imp/U_{U:.1f}", Us, "Spectral function vs U (Ng=4)",
             "scan_ng4_not_imp/eqp_vs_U_3d_pos.png", omega_min=0.0)
p.ridge_plot(lambda U: f"scan_ng4_dV2_confirmed/U_{U:.1f}", Us, "Ng=4, dV=2",
             "scan_ng4_dV2_confirmed/eqp_vs_U_3d_pos.png", ed_fn=q.ed_Aomega_dV2, omega_min=0.0)
