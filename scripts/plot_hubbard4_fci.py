"""PNG plots of the 4-site Hubbard FCI spectral functions written by hubbard4_fci.py.

    python scripts/plot_hubbard4_fci.py [--U 10] [--dir hubbard4_fci]

For the chosen U: local/total A(w) for OBC and PBC, A_orb (OBC), A_k (PBC).
Also A_tot(w) vs U colour maps for both boundary conditions.
Output in <dir>/plots/.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--U", type=float, nargs="+", default=[10.0])
ap.add_argument("--dir", default="hubbard4_fci")
args = ap.parse_args()

pdir = os.path.join(args.dir, "plots")
os.makedirs(pdir, exist_ok=True)
BCS = [b for b in ("obc", "pbc", "apbc") if os.path.exists(os.path.join(args.dir, b, "spectra.npz"))]
S = {bc: np.load(os.path.join(args.dir, bc, "spectra.npz")) for bc in BCS}
w = S["obc"]["w"]
Us = S["obc"]["U"]
KLAB = dict(pbc=["0", r"\pi/2", r"\pi", r"3\pi/2"], apbc=[r"\pi/4", r"3\pi/4", r"5\pi/4", r"7\pi/4"])

for U in args.U:
    iu = int(np.argmin(np.abs(Us - U)))
    U = Us[iu]
    fig, axs = plt.subplots(2, len(BCS), figsize=(6 * len(BCS), 8), sharex=True)
    for col, bc in enumerate(BCS):
        s = S[bc]
        ax = axs[0, col]
        for i in range(4):
            ax.plot(w, s["A_loc"][iu, i], lw=1, label=f"site {i + 1}")
        ax.plot(w, s["A_tot"][iu], "k", lw=1.5, label="total")
        ax.set_title(f"{bc.upper()}  U={U:g}: local spectral function")
        ax.set_ylabel(r"$A(\omega)$ per spin")
        ax.legend(fontsize=8)
        ax = axs[1, col]
        if "A_k" in s.files:
            for k in range(4):
                ax.plot(w, s["A_k"][iu, k], lw=1.2, label=rf"$k={KLAB[bc][k]}$")
            ax.set_title(f"{bc.upper()}  U={U:g}: momentum-resolved $A_k(\\omega)$")
        else:
            for a in range(4):
                ax.plot(w, s["A_orb"][iu, a], lw=1.2,
                        label=rf"orb {a + 1} ($\epsilon_0$={s['eps0'][a]:+.3f})")
            ax.set_title(f"OBC  U={U:g}: A in U=0 orbital basis")
        ax.set_xlabel(r"$\omega-\mu$")
        ax.set_ylabel(r"$A(\omega)$ per spin")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fn = os.path.join(pdir, f"A_U{U:05.2f}.png")
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("wrote", fn)

fig, axs = plt.subplots(1, len(BCS), figsize=(6 * len(BCS), 5), sharey=True)
for ax, bc in zip(axs, BCS):
    s = S[bc]
    im = ax.pcolormesh(Us, w, np.log10(s["A_tot"].T + 1e-4), shading="auto", cmap="magma",
                       vmin=-3, vmax=1)
    ax.set_title(f"{bc.upper()}: $\\log_{{10}} A_{{tot}}(\\omega)$")
    ax.set_xlabel("U")
axs[0].set_ylabel(r"$\omega-\mu$")
fig.colorbar(im, ax=axs)
fn = os.path.join(pdir, "A_tot_vs_U.png")
fig.savefig(fn, dpi=150)
plt.close(fig)
print("wrote", fn)
