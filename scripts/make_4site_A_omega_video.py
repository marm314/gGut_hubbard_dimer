"""Animations of A(omega) vs U/t for the half-filled 4-site Hubbard model, one frame per U
(0..20, step 0.2), same style as make_A_omega_video.py for the dimer: (g)Gutzwiller A_tot(w)
of the lowest-energy branch (filled) vs exact FCI A_tot(w) (red line), per spin, eta = 0.05,
w relative to mu = U/2.

    ~/.venv/bin/python scripts/make_4site_A_omega_video.py [--Ng 0 2] [--bc obc pbc apbc]

Writes scan_4site_ng{Ng}/{bc}/A_omega_vs_U.mp4 (ffmpeg from imageio-ffmpeg).
"""
import argparse
import os

import numpy as np
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WMAX = 14.0
NAME = dict(obc="OBC chain", pbc="PBC ring", apbc="APBC ring")


def lowest_runs(ng, bc):
    """{U: (run dir, E_var, |R_1|^2)} of the lowest-energy branch."""
    base = os.path.join(REPO, f"scan_4site_ng{ng}", bc)
    out = {}
    for l in open(os.path.join(base, "lowest_energy.txt")):
        if l.startswith("#"):
            continue
        p = l.split()
        U = round(float(p[0]), 6)
        out[U] = (os.path.join(base, p[-1], f"U_{U:05.2f}"), float(p[1]), float(p[3]) ** 2)
    return out


def make_video(ng, bc, fps=10):
    fci = np.load(os.path.join(REPO, "hubbard4_fci", bc, "spectra.npz"))
    runs = lowest_runs(ng, bc)
    m = np.abs(fci["w"]) <= WMAX
    w = fci["w"][m]
    frames = []
    for iu, U in enumerate(np.round(fci["U"], 6)):
        if U not in runs:
            continue
        d, E, Z = runs[U]
        A = np.load(os.path.join(d, "A.npz"))
        Ag = np.einsum("wii->w", A["A"])[np.abs(A["w"]) <= WMAX]
        Ef = fci_E[bc][iu] - 2 * U
        frames.append((U, Ag, fci["A_tot"][iu][m], E - Ef, Z))
    zmax = max(max(f[1].max(), f[2].max()) for f in frames)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-WMAX, WMAX)
    ax.set_ylim(0, 1.05 * zmax)
    ax.set_xlabel(r"$\omega-\mu$ / t")
    ax.set_ylabel(r"$A_{tot}(\omega)$ per spin  ($\eta$=0.05)")
    fill = ax.fill_between(w, frames[0][1], color="teal", alpha=0.75)
    line_fci, = ax.plot(w, frames[0][2], color="red", lw=1.2, label="FCI (exact)")
    line_g, = ax.plot(w, frames[0][1], color="darkslategray", lw=0.9,
                      label=f"Gutzwiller $N_g$={ng} (lowest branch)")
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, va="top")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"4-site Hubbard, half filling, {NAME[bc]}: $N_g$={ng} vs FCI"
                 + ("\n(PBC Gutzwiller unreliable: degenerate Fermi level)" if bc == "pbc" else ""))
    plt.tight_layout()

    def update(i):
        nonlocal fill
        U, Ag, Af, dE, Z = frames[i]
        fill.remove()
        fill = ax.fill_between(w, Ag, color="teal", alpha=0.75)
        line_g.set_data(w, Ag)
        line_fci.set_data(w, Af)
        txt.set_text(f"U/t = {U:.1f}\n$E_{{var}}-E_{{FCI}}$ = {dE:.3f}\n$|R_1|^2$ = {Z:.3f}")
        return fill, line_g, line_fci, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = os.path.join(REPO, f"scan_4site_ng{ng}", bc, "A_omega_vs_U.mp4")
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    print("wrote", out, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--Ng", type=int, nargs="+", default=[0, 2])
ap.add_argument("--bc", nargs="+", default=["obc", "pbc", "apbc"])
args = ap.parse_args()
fci_E = {bc: np.loadtxt(os.path.join(REPO, "hubbard4_fci", bc, "energies.dat"))[:, 1] for bc in args.bc}
for ng in args.Ng:
    for bc in args.bc:
        make_video(ng, bc)
