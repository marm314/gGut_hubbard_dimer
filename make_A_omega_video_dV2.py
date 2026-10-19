"""Delta V/t=2.0 analogue of make_A_omega_video.py, for scan_ng2_imp_dV2/,
scan_ng2_not_imp_dV2/ and scan_ng4_not_imp_dV2/ (see scan_ng2_imp_dV2_U.py,
scan_ng2_not_imp_dV2_U.py, scan_ng4_not_imp_dV2_U.py). Same quantities as
plot_eqp_3d_dV2.py (gGA A(omega), filled; exact ED A(omega) from the
scan_ed_deltaV/ED_deltaV.tex closed form at Delta V=2, black line; same eta), but one
frame per U-point on the full 0.1 .. 10.0 grid.

Requires ffmpeg (found at /usr/bin/ffmpeg); writes <scan_dir>/A_omega_vs_U.mp4.
"""
import numpy as np

from make_A_omega_video import make_video
from plot_eqp_3d_dV2 import ed_Aomega_dV2, load_branches

Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]

if __name__ == "__main__":
    # 1) scan_ng2_imp_dV2
    br1 = load_branches("scan_ng2_imp_dV2/lowest_energy.txt")
    make_video(lambda U: f"scan_ng2_imp_dV2/{br1[U]}/U_{U:.1f}",
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$)",
               "scan_ng2_imp_dV2/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)

    # 2) scan_ng2_not_imp_dV2
    br2 = load_branches("scan_ng2_not_imp_dV2/lowest_energy.txt")
    make_video(lambda U: f"scan_ng2_not_imp_dV2/{br2[U]}/U_{U:.1f}",
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$)",
               "scan_ng2_not_imp_dV2/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)

    # 3) scan_ng4_not_imp_dV2
    make_video(lambda U: f"scan_ng4_not_imp_dV2/U_{U:.1f}",
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$)",
               "scan_ng4_not_imp_dV2/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)

    # 4) scan_ng4_imp_dV2
    br4 = load_branches("scan_ng4_imp_dV2/lowest_energy.txt")
    make_video(lambda U: f"scan_ng4_imp_dV2/{br4[U]}/U_{U:.1f}",
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$)",
               "scan_ng4_imp_dV2/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)

    # 5) scan_ng4_dV2_confirmed: cross_check_ng4_dV2.py's mutually-confirmed winner per U
    make_video(lambda U: f"scan_ng4_dV2_confirmed/U_{U:.1f}",
               r"Spectral function vs U (Ng=4, $\Delta V/t=2$, cross-checked mutual solution)",
               "scan_ng4_dV2_confirmed/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)

    # 6) scan_ng2_dV2_confirmed: cross_check_ng2_dV2.py's mutually-confirmed winner per U
    make_video(lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}",
               r"Spectral function vs U (Ng=2, $\Delta V/t=2$, cross-checked mutual solution)",
               "scan_ng2_dV2_confirmed/A_omega_vs_U.mp4", ed_fn=ed_Aomega_dV2)
