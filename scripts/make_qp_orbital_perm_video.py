"""qp orbitals, PERMUTATIONS ONLY: raw eigh eigenvectors in the solver chi basis (energy order, no orbital tracking, no mixing), with only the order of the chi_i inside each site changed to
follow the previous U/t (see make_qp_orbital_videos.perm_only_frames). usage (repo root): python3 scripts/make_qp_orbital_perm_video.py dV0|dV2  -> <scan>/qp_orbitals_vs_U.mp4"""
import sys
sys.path.insert(0, "scripts")
import matplotlib; matplotlib.use("Agg")
import make_qp_orbital_videos as mq
which = sys.argv[1] if len(sys.argv) > 1 else "dV0"
if which == "dV0":
    br = {}
    for l in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not l.startswith("#") and l.strip():
            f = l.split(); br[round(float(f[0]), 1)] = f[2]
    d, get, ttl = "scan_ng2_not_imp", (lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}"), "Ng=2, dV=0"
else:
    d, get, ttl = "scan_ng2_dV2_confirmed", (lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}"), "Ng=2, dV=2"
rf = mq.perm_only_frames(mq.load_frames(get))
mq.make_orbitals(rf, ttl, f"{d}/qp_orbitals_vs_U.mp4", key="phx")
