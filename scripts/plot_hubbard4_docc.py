"""Double occupancy D_i = <n_iup n_idn> of the half-filled 4-site Hubbard model (FCI),
OBC and PBC, vs U/t on the hubbard4_fci.py grid. Degenerate ground states (PBC, U=0)
are ensemble-averaged. Checked against Hellmann-Feynman, D = (1/L) dE0/dU.

    ~/.venv/bin/python scripts/plot_hubbard4_docc.py [--dir hubbard4_fci] [--mark 1 10]

Writes <dir>/{obc,pbc,apbc}/docc.dat and <dir>/plots/docc_vs_U.png.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf.fci import direct_spin1

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hubbard4_fci import L, NHALF, all_sectors, hopping, lowest, onsite_eri

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="hubbard4_fci")
ap.add_argument("--t", type=float, default=1.0)
ap.add_argument("--mark", type=float, nargs="+", default=[1.0, 10.0])
args = ap.parse_args()


def docc(U, bc, degtol=1e-8):
    sec = all_sectors(hopping(args.t, bc), onsite_eri(U), NHALF)
    E0 = lowest(sec)
    D, ng = np.zeros(L), 0
    for nel, (E, C) in sec.items():
        for n in np.where(E - E0 < degtol)[0]:
            (_, _), (_, dm2ab, _) = direct_spin1.make_rdm12s(C[n], L, nel)
            D += np.array([dm2ab[i, i, i, i] for i in range(L)])
            ng += 1
    return E0, D / ng


fig, ax = plt.subplots(figsize=(7, 5))
for bc, c in (("obc", "C0"), ("pbc", "C3"), ("apbc", "C1")):
    if not os.path.exists(os.path.join(args.dir, bc, "energies.dat")):
        continue
    Us = np.loadtxt(os.path.join(args.dir, bc, "energies.dat"))[:, 0]
    res = [docc(U, bc) for U in Us]
    E0 = np.array([r[0] for r in res])
    D = np.array([r[1] for r in res])
    Dav = D.mean(axis=1)
    hf = np.gradient(E0, Us) / L
    print(f"{bc}: max |D - dE0/dU/L| (finite diff) = {np.abs(Dav - hf)[1:-1].max():.2e}")
    np.savetxt(os.path.join(args.dir, bc, "docc.dat"), np.column_stack([Us, Dav, D]),
               header=f"{bc.upper()} t={args.t}\nU/t  D_avg  D_1  D_2  D_3  D_4", fmt="%.10f")
    p = Us > 0  # U=0 left out of the plot (degenerate PBC ensemble value); kept in docc.dat
    ax.plot(Us[p] / args.t, Dav[p], "-", color=c, label=f"{bc.upper()} (site average)")
    if bc == "obc":
        ax.plot(Us[p] / args.t, D[p, 0], ":", color=c, lw=1, label="OBC edge site")
        ax.plot(Us[p] / args.t, D[p, 1], "--", color=c, lw=1, label="OBC bulk site")
    for Um in args.mark:
        i = int(np.argmin(np.abs(Us - Um)))
        ax.plot(Us[i] / args.t, Dav[i], "o", color=c)
        ax.annotate(f"{Dav[i]:.4f}", (Us[i] / args.t, Dav[i]), textcoords="offset points",
                    xytext=(6, {"obc": -14, "pbc": 6, "apbc": 14}[bc]), color=c, fontsize=9)
        print(f"  U/t={Us[i] / args.t:5.2f}  D_avg={Dav[i]:.6f}  D_sites={np.round(D[i], 6)}")

ax.axhline(0.25, color="gray", lw=0.8, ls=":")
ax.text(Us[-1] / args.t - 0.1, 0.252, "uncorrelated 1/4", ha="right", va="bottom", color="gray", fontsize=8)
ax.set_xlabel("U/t")
ax.set_ylabel(r"$D = \langle n_\uparrow n_\downarrow \rangle$ per site")
ax.set_title("4-site Hubbard, half filling, FCI: double occupancy")
ax.set_xlim(0, Us[-1] / args.t)
ax.set_ylim(0, 0.27)
ax.legend()
fig.tight_layout()
os.makedirs(os.path.join(args.dir, "plots"), exist_ok=True)
fn = os.path.join(args.dir, "plots", "docc_vs_U.png")
fig.savefig(fn, dpi=150)
print("wrote", fn)
