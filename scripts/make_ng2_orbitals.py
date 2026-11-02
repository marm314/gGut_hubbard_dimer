"""Re-render only the Ng=2 orbital movies (dV=0 and dV=2) with the orbital tracking of make_qp_orbital_videos.align_frames. usage (repo root): python3 scripts/make_ng2_orbitals.py"""
import sys
sys.path.insert(0, "scripts")
import make_qp_orbital_videos as mo
br2 = {}
for line in open("scan_ng2_not_imp/lowest_energy.txt"):
    if not line.startswith("#") and line.strip(): f = line.split(); br2[round(float(f[0]), 1)] = f[2]
for d, get, ttl in (("scan_ng2_not_imp", lambda U: f"scan_ng2_not_imp/{br2[U]}/U_{U:.1f}", "Ng=2, dV=0"), ("scan_ng2_dV2_confirmed", lambda U: f"scan_ng2_dV2_confirmed/U_{U:.1f}", "Ng=2, dV=2")):
    mo.make_orbitals(mo.load_frames(get), ttl, f"{d}/qp_orbitals_vs_U.mp4")
