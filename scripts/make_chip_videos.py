"""chi' (star basis) movies, Ng=2: qp_orbitals_vs_U_chip.mp4 and qp_density_vs_U_chip.mp4 (the physical-orbital one: make_phys_orbital_videos.py dV0|dV2 chip).
chi' = the basis obtained from the solver chi by the rotation U_s of each site whose first column is R_s/|R_s| (so R'_s = (|R_s|,0,..): chi'_1 = state) and whose other columns diagonalize -lambda_s
projected orthogonal to R_s (ghosts) (Lanata, arXiv:2305.11895, Sec. III). Orbitals are followed in U/t (make_qp_orbital_videos.load_frames: tracked, phases aligned).
usage (repo root): python3 scripts/make_chip_videos.py dV0|dV2 [orbitals|density]"""
import sys
sys.path.insert(0, "scripts")
import matplotlib; matplotlib.use("Agg")
import make_qp_orbital_videos as mq
which = sys.argv[1] if len(sys.argv) > 1 else "dV0"
what = sys.argv[2:] or ["orbitals", "density"]
if which == "dV0":
    br = {}
    for l in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not l.startswith("#") and l.strip():
            f = l.split(); br[round(float(f[0]), 1)] = f[2]
    d, get, ttl = "scan_ng2_not_imp", (lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}"), "Ng=2, dV=0"
else:
    d, get, ttl = "scan_ng2_dV2_confirmed", (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}"), "Ng=2, dV=2"
fr = mq.load_frames(get)
if "orbitals" in what: mq.make_orbitals(fr, ttl, f"{d}/qp_orbitals_vs_U_chip.mp4", key="ph")
if "density" in what: mq.make_density(fr, ttl, f"{d}/qp_density_vs_U_chip.mp4", basis="chip")
