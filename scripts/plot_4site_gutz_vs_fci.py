"""Gutzwiller (C++ run_ggut, scan_4site_cpp.py) vs FCI (hubbard4_fci.py) for the
half-filled 4-site Hubbard model: energy error, double occupancy and R vs U/t,
for the lowest-energy branch per U (lowest_energy.txt; forward/backward runs stay on disk).

    ~/.venv/bin/python scripts/plot_4site_gutz_vs_fci.py [--Ng 0]

Writes scan_4site_ng{Ng}/gutz_vs_fci.png.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--Ng", type=int, default=0)
args = ap.parse_args()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
top = os.path.join(REPO, f"scan_4site_ng{args.Ng}")

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
BCS = [b for b in ("obc", "pbc", "apbc") if os.path.exists(os.path.join(top, b, "summary_forward.txt"))]
COL = dict(obc="C0", pbc="C3", apbc="C1")
for bc in BCS:
    c = COL[bc]
    fci = np.loadtxt(os.path.join(REPO, "hubbard4_fci", bc, "energies.dat"))
    dfci = np.loadtxt(os.path.join(REPO, "hubbard4_fci", bc, "docc.dat"))
    p = dfci[:, 0] > 0
    efci = dict(zip(np.round(fci[:, 0], 6), fci[:, 1] - 2 * fci[:, 0]))  # FCI E - (U/2)N
    axs[1].plot(dfci[p, 0], dfci[p, 1], "-", color=c, label=f"{bc.upper()} FCI")
    # lowest-energy converged branch per U (forward/backward runs are kept on disk)
    g = np.loadtxt(os.path.join(top, bc, "lowest_energy.txt"), usecols=range(9))
    lab = f"{bc.upper()} Gutzwiller Ng={args.Ng}"
    err = g[:, 1] - np.array([efci[u] for u in np.round(g[:, 0], 6)])
    axs[0].plot(g[:, 0], err, "o", color=c, ms=4, mfc="none", label=lab)
    q = g[:, 0] > 0
    axs[1].plot(g[q, 0], g[q, 2], "o", color=c, ms=4, mfc="none", label=lab)
    axs[2].plot(g[:, 0], g[:, 3] ** 2, "o", color=c, ms=4, mfc="none",
                label=f"{lab}, site 1" if bc == "obc" else lab)
    if bc == "obc":
        axs[2].plot(g[:, 0], g[:, 4] ** 2, "s", color=c, ms=3, mfc="none", label=f"{lab}, site 2")
axs[0].set_ylabel(r"$E_{var} - E_{FCI}$ / t")
axs[0].axhline(0, color="gray", ls=":", lw=0.8)
axs[1].set_ylabel(r"$D=\langle n_\uparrow n_\downarrow\rangle$ per site")
axs[1].axhline(0.25, color="gray", ls=":", lw=0.8)
axs[2].set_ylabel(r"$Z = R^2$")
for ax, t in zip(axs, ("Energy error vs FCI", "Double occupancy", "Quasiparticle weight")):
    ax.set_xlabel("U/t")
    ax.set_title(t)
    ax.legend(fontsize=7)
fig.suptitle(f"4-site Hubbard, half filling: Gutzwiller (Ng={args.Ng}, C++, lowest of forward/backward/polish) vs FCI")
fig.tight_layout()
fn = os.path.join(top, "gutz_vs_fci.png")
fig.savefig(fn, dpi=150)
print("wrote", fn)
for bc in BCS:
    fci = np.loadtxt(os.path.join(REPO, "hubbard4_fci", bc, "energies.dat"))
    g = np.loadtxt(os.path.join(top, bc, "lowest_energy.txt"), usecols=range(9))
    d = g[:, 1] - (fci[:, 1] - 2 * fci[:, 0])
    print(f"{bc}: min(E_gutz - E_fci) = {d.min():+.5f} at U={g[d.argmin(), 0]}")
