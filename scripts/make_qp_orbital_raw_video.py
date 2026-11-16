"""qp orbitals in the raw chi basis of the solver: phi^qp_n = sum_i chi_i (C_qp)_{in} with C_qp the eigenvectors returned by eigh for H_qp built from the saved R, lambda of each U/t;
no basis rotation, no orbital tracking (energy order as eigh); only the overall sign of each orbital is fixed (largest component positive).
usage (repo root): python3 scripts/make_qp_orbital_raw_video.py  -> scan_ng2_not_imp/qp_orbitals_chi_raw_vs_U.mp4"""
import sys
sys.path.insert(0, "scripts")
import matplotlib; matplotlib.use("Agg")
import make_qp_orbital_videos as mq
br = {}
for l in open("scan_ng2_not_imp/lowest_energy.txt"):
    if not l.startswith("#") and l.strip():
        f = l.split(); br[round(float(f[0]), 1)] = f[2]
fr = mq.load_frames(lambda U: f"scan_ng2_not_imp/{br[U]}/U_{U:.1f}")
rf = [dict(f, **f["raw"]) for f in fr]
mq.make_orbitals(rf, "Ng=2, dV=0, raw solver basis", "scan_ng2_not_imp/qp_orbitals_chi_raw_vs_U.mp4", key="phx", mode="raw")
