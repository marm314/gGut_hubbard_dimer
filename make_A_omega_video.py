"""Animations of the evolution of A(omega) with U/t: one frame per U-point (the full, fine
0.1 .. 10.0 grid, not just the coarse slices of the eqp_vs_U_3d.png ridgeline), same
quantities as plot_eqp_3d.py (gGA A(omega), filled; exact ED A(omega), red line; same eta).

Requires ffmpeg (found at /usr/bin/ffmpeg); writes <scan_dir>/A_omega_vs_U.mp4.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gga_dimer as g
from plot_eqp_3d import ed_Aomega, load_R_lambda, branch_of, t, ETA, OMEGA_MAX, DOMEGA

Us = [round(u, 1) for u in np.arange(0.1, 10.01, 0.1)]


def make_video(get_dir_for_U, title, out_path, fps=12, ed_fn=ed_Aomega):
    """ed_fn(omega, U, t, eta): the exact-ED A(omega) overlay; defaults to the Delta V=0
    closed form of plot_eqp_3d.py, but e.g. make_A_omega_video_dV2.py passes a Delta V=2
    one instead."""
    # precompute all frames and the global y-limit
    frames = []
    zmax = 0.0
    for U in Us:
        R, lam = load_R_lambda(get_dir_for_U(U))
        omega_grid, A = g.compute_spectral_function(R[0], R[1], lam[0], lam[1], t, OMEGA_MAX, ETA, DOMEGA)
        A_ed = ed_fn(omega_grid, U, t, ETA)
        zmax = max(zmax, A.max(), A_ed.max())
        frames.append((omega_grid, A, A_ed))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-OMEGA_MAX, OMEGA_MAX)
    ax.set_ylim(0, 1.08 * zmax)
    ax.set_xlabel("quasiparticle energy / t")
    ax.set_ylabel(rf"A($\omega$)  ($\eta$={ETA:g})")
    fill = ax.fill_between(frames[0][0], frames[0][1], color="teal", alpha=0.75)
    line_ed, = ax.plot(frames[0][0], frames[0][2], color="red", lw=1.3, label="ED A(omega) (exact)")
    line_gga, = ax.plot(frames[0][0], frames[0][1], color="darkslategray", lw=1.0, label="gGA A(omega)")
    txt = ax.text(0.03, 0.92, "", transform=ax.transAxes, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)
    plt.tight_layout()

    def update(i):
        omega_grid, A, A_ed = frames[i]
        nonlocal fill
        fill.remove()
        fill = ax.fill_between(omega_grid, A, color="teal", alpha=0.75)
        line_gga.set_data(omega_grid, A)
        line_ed.set_data(omega_grid, A_ed)
        txt.set_text(f"U/t = {Us[i]:.1f}")
        return fill, line_gga, line_ed, txt

    ani = animation.FuncAnimation(fig, update, frames=len(Us), blit=False)
    ani.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print(f"wrote {out_path}  ({len(Us)} frames @ {fps} fps = {len(Us)/fps:.1f}s)")


if __name__ == "__main__":
    # 1) scan_ng2_imp
    br = branch_of("scan_ng2_imp/backward", "scan_ng2_imp/forward")
    make_video(lambda U: br[U] + f"/U_{U:.1f}",
               "Spectral function vs U (Ng=2, minimum-energy branch)",
               "scan_ng2_imp/A_omega_vs_U.mp4")

    # 2) scan_ng2_not_imp
    br2 = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip():
            p = line.split()
            br2[round(float(p[0]), 1)] = p[2]
    make_video(lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}",
               "Spectral function, lowest-energy solution (Ng=2)",
               "scan_ng2_not_imp/A_omega_vs_U.mp4")

    # 3) scan_ng2_not_imp_sol_at_imp
    make_video(lambda U: f"scan_ng2_not_imp_sol_at_imp/U_{U:.1f}",
               "Spectral function of gga_dimer.py (Ng=2),\nseeded with the direct-minimization solutions",
               "scan_ng2_not_imp_sol_at_imp/A_omega_vs_U.mp4")

    # 4) scan_ng4_not_imp
    make_video(lambda U: f"scan_ng4_not_imp/U_{U:.1f}",
               "Spectral function vs U (Ng=4)",
               "scan_ng4_not_imp/A_omega_vs_U.mp4")
