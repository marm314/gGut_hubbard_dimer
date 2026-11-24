"""Spectral functions of the half-filled 4-site Hubbard model: FCI (hubbard4_fci.py) vs
(g)Gutzwiller Ng=0 and Ng=2 (C++ run_ggut, lowest-energy branch per U from lowest_energy.txt).

Both are per spin, Lorentzian eta = 0.05, w relative to mu = U/2.

    ~/.venv/bin/python scripts/plot_4site_spectra_gutz_vs_fci.py [--U 1 5 10 20]

Writes hubbard4_fci/plots/spectra_gutz_vs_fci.png (A_tot = sum_i A_ii, rows U, columns bc)
and spectra_k_gutz_vs_fci_apbc.png (k-resolved A_k, antiperiodic ring).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 4
ap = argparse.ArgumentParser()
ap.add_argument("--U", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
ap.add_argument("--wmax", type=float, default=14.0)
args = ap.parse_args()
BCS = ("obc", "pbc", "apbc")
NGS = ((0, "C0"), (2, "C3"))


def gutz_A(ng, bc, U):
    """A_ij(w) of the lowest-energy run at U (dir from the branch column), or None."""
    base = os.path.join(REPO, f"scan_4site_ng{ng}", bc)
    for l in open(os.path.join(base, "lowest_energy.txt")):
        p = l.split()
        if l.startswith("#") or abs(float(p[0]) - U) > 1e-6:
            continue
        d = np.load(os.path.join(base, p[-1], f"U_{U:05.2f}", "A.npz"))
        return d["w"], d["A"], float(p[1])
    return None


FCI = {bc: np.load(os.path.join(REPO, "hubbard4_fci", bc, "spectra.npz")) for bc in BCS}
pdir = os.path.join(REPO, "hubbard4_fci", "plots")

# --- local (site-summed) spectra
fig, axs = plt.subplots(len(args.U), len(BCS), figsize=(5.5 * len(BCS), 2.6 * len(args.U)),
                        sharex=True, squeeze=False)
for r, U in enumerate(args.U):
    for c, bc in enumerate(BCS):
        ax = axs[r, c]
        f = FCI[bc]
        iu = int(np.argmin(np.abs(f["U"] - U)))
        ax.plot(f["w"], f["A_tot"][iu], "k", lw=1.4, label="FCI")
        for ng, col in NGS:
            g = gutz_A(ng, bc, U)
            if g is None:
                continue
            w, A, _ = g
            ax.plot(w, np.einsum("wii->w", A), color=col, lw=1, label=f"Gutzwiller Ng={ng}")
        ax.set_title(f"{bc.upper()}  U/t={U:g}" + ("  (Gutzwiller unreliable)" if bc == "pbc" else ""),
                     fontsize=10)
        ax.set_xlim(-args.wmax, args.wmax)
        if c == 0:
            ax.set_ylabel(r"$A_{tot}(\omega)$ per spin")
        if r == 0 and c == 0:
            ax.legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel(r"$\omega-\mu$ / t")
fig.suptitle("4-site Hubbard, half filling: total spectral function, FCI vs Gutzwiller (lowest branch)")
fig.tight_layout()
fn = os.path.join(pdir, "spectra_gutz_vs_fci.png")
fig.savefig(fn, dpi=130)
plt.close(fig)
print("wrote", fn)

# --- k-resolved, antiperiodic ring: k = (2n+1) pi / 4, c_k = 1/sqrt(L) sum_j e^{-ikj} c_j
ks = (2 * np.arange(L) + 1) * np.pi / L
Fk = np.exp(1j * np.outer(ks, np.arange(L))) / np.sqrt(L)
klab = [r"\pi/4", r"3\pi/4", r"5\pi/4", r"7\pi/4"]
f = FCI["apbc"]
fig, axs = plt.subplots(len(args.U), 2, figsize=(12, 2.6 * len(args.U)), sharex=True, squeeze=False)
for r, U in enumerate(args.U):
    iu = int(np.argmin(np.abs(f["U"] - U)))
    for c, kk in enumerate((0, 1)):  # pi/4 (= 7pi/4) and 3pi/4 (= 5pi/4) by symmetry
        ax = axs[r, c]
        ax.plot(f["w"], f["A_k"][iu, kk], "k", lw=1.4, label="FCI")
        for ng, col in NGS:
            g = gutz_A(ng, "apbc", U)
            if g is None:
                continue
            w, A, _ = g
            Ak = np.einsum("i,wij,j->w", Fk[kk], A, Fk[kk].conj()).real
            ax.plot(w, Ak, color=col, lw=1, label=f"Gutzwiller Ng={ng}")
        ax.set_title(rf"APBC  U/t={U:g}  $k={klab[kk]}$", fontsize=10)
        ax.set_xlim(-args.wmax, args.wmax)
        if r == 0 and c == 0:
            ax.legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel(r"$\omega-\mu$ / t")
fig.suptitle(r"Antiperiodic ring: $A_k(\omega)$, FCI vs Gutzwiller (lowest branch)")
fig.tight_layout()
fn = os.path.join(pdir, "spectra_k_gutz_vs_fci_apbc.png")
fig.savefig(fn, dpi=130)
plt.close(fig)
print("wrote", fn)
