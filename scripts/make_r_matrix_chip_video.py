"""R matrix in the chi' (star) basis vs U/t, Ng=2: <scan_dir>/r_matrix_in_chip.mp4.
R' = Rrow T^T R (2 x 2(1+Ng); row s = site s, columns chi'_1..chi'_{2(1+Ng)}), computed numerically from the saved R and the star rotation T of make_qp_orbital_videos (Rrow = the re-identification of the ghosts
done when following the orbitals in U/t, the same chi' as in the *_chip.mp4 movies). Heatmap of R'_{sj} with the entries, and |R_1|, |R_2|, Z vs U/t.
usage (repo root): python3 scripts/make_r_matrix_chip_video.py dV0|dV2"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import make_qp_orbital_videos as mq

which = sys.argv[1] if len(sys.argv) > 1 else "dV0"
if which == "dV0":
    br = {}
    for l in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not l.startswith("#") and l.strip():
            f = l.split(); br[round(float(f[0]), 1)] = f[2]
    d, get, ttl = "scan_ng2_not_imp", (lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}"), r"$N_g=2$, $\Delta V=0$"
else:
    d, get, ttl = "scan_ng2_dV2_confirmed", (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}"), r"$N_g=2$, $\Delta V=2$"
fr = mq.load_frames(get); Neff = fr[0]["Neff"]; N = 2 * Neff
Rp = []
for f in fr:
    Rm = np.zeros((N, 2)); Rm[:Neff, 0] = f["R"][0]; Rm[Neff:, 1] = f["R"][1]
    Rp.append((f["Rrow"] @ (f["T"].T @ Rm)).T)                 # (2, N): R'_{sj}
Rp = np.array(Rp); Us = np.array([f["U"] for f in fr]); Zs = np.array([f["z"] for f in fr])
nR = np.array([[np.linalg.norm(f["R"][s]) for s in range(2)] for f in fr])
st = [0, Neff]                                                   # column of the state chi'_1 of each site
print("largest ghost entry of R' over all U/t:", max(np.abs(np.delete(Rp[k, s], st[s])).max() for k in range(len(fr)) for s in range(2)), flush=True)
fig = plt.figure(figsize=(12.5, 5.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1])
ax = fig.add_subplot(gs[:, 0]); axp = fig.add_subplot(gs[0, 1]); axz = fig.add_subplot(gs[1, 1])
axp.plot(Us, nR[:, 0], label=r"$|R_1|$"); axp.plot(Us, nR[:, 1], ls="--", label=r"$|R_2|$"); axp.legend(fontsize=9); axp.set_ylabel(r"$|R_s|$")
axz.plot(Us, nR[:, 0] ** 2, label=r"$|R_1|^2$"); axz.plot(Us, nR[:, 1] ** 2, ls="--", label=r"$|R_2|^2$"); axz.plot(Us, Zs, color="C3", label="Z"); axz.legend(fontsize=9); axz.set_xlabel("U/t")
marks = [a.axvline(Us[0], color="k", lw=1) for a in (axp, axz)]
txt = fig.suptitle("")
im = ax.imshow(np.abs(Rp[0]), vmin=0, vmax=1, cmap="viridis", aspect="equal")
fig.colorbar(im, ax=ax, fraction=0.035, label=r"$|R'_{sj}|$")
ax.set_xticks(range(N)); ax.set_xticklabels([rf"$\chi'_{{{j + 1}}}$" for j in range(N)], fontsize=11); ax.set_yticks(range(2)); ax.set_yticklabels([r"$s=1$", r"$s=2$"], fontsize=11)
ax.axvline(Neff - 0.5, color="w", lw=2)
labs = [[ax.text(j, s, "", ha="center", va="center", fontsize=10) for j in range(N)] for s in range(2)]
ax.set_title(r"$R'_{sj}=\sum_iR_{si}(U_s)_{ij}$   (columns: site 1 | site 2; $\chi'_1,\chi'_{%d}$ = states)" % (Neff + 1), fontsize=10)
def update(i):
    M = Rp[i]; im.set_data(np.abs(M))
    for s in range(2):
        for j in range(N):
            v = M[s, j]; labs[s][j].set_text("0" if abs(v) < 1e-9 else f"{v:+.3f}"); labs[s][j].set_color("w" if abs(v) < 0.5 else "k")
    for m_ in marks: m_.set_xdata([Us[i], Us[i]])
    txt.set_text(rf"{ttl}   $U/t$ = {Us[i]:.1f}   $|R_1|$ = {nR[i, 0]:.4f}   $|R_2|$ = {nR[i, 1]:.4f}   $Z$ = {Zs[i]:.4f}")
    return []
ani = animation.FuncAnimation(fig, update, frames=len(fr), blit=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
ani.save(f"{d}/r_matrix_in_chip.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=2500)); plt.close(fig)
print("wrote", f"{d}/r_matrix_in_chip.mp4", flush=True)
