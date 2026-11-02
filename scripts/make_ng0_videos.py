"""Ng=0 movies (same scripts and layout as the Ng=2 ones), from the saved Ng=0 scans (scan_ng0/, lowest-energy branch): for dV=0 and dV=2 writes in scan_ng0/dV_<dV>/
A_omega_vs_U.mp4, qp_weights_vs_U.mp4, qp_orbitals_vs_U.mp4, qp_density_vs_U.mp4. usage (repo root): python3 scripts/make_ng0_videos.py [A|weights|orbitals|density ...]"""
import sys
sys.path.insert(0, "scripts")
import plot_eqp_3d_ng0 as ng0           # patches plot_eqp_3d.load_R_lambda with the Ng=0 loader (key "dV|U")
import plot_eqp_3d as p
import plot_eqp_3d_dV2 as q
import make_A_omega_video as mav
import make_qp_weights_video as mw
import make_qp_orbital_videos as mo
import os, re, numpy as np
from ng0_analytic import ng0_state
def adir(dV, U):                       # directory of the analytic solution at (dV, U): output.out with Z0, docc (read by make_qp_orbital_videos.load_frames)
    d = f"scan_ng0/dV_{dV:.1f}/analytic/U_{U:.1f}"; os.makedirs(d, exist_ok=True); a = ng0_state(U, dV)
    open(d + "/output.out", "w").write(f"Ng=0  U/t={U:.4f}  deltaV/t={dV:.4f}  Z0={a['Z']:.8f}  Z1={a['Z']:.8f}  E_var/t={a['E']:.8f}  docc={a['docc']:.8f}  (analytic)\n"); return d
def load_path(d):                      # analytic R and lambda from the directory name
    m = re.search(r"dV_([\d.]+)/analytic/U_([\d.]+)", d); return ng0.load_ng0(f"{m.group(1)}|{m.group(2)}")
for m in (mav, mw, mo): m.load_R_lambda = load_path
what = sys.argv[1:] or ["A", "weights", "orbitals", "density"]
for dV, ed_A, ed_p in ((0.0, p.ed_Aomega, mw.ed_poles0), (2.0, q.ed_Aomega_dV2, mw.ed_poles2)):
    d = f"scan_ng0/dV_{dV:.1f}"; get = lambda U, dV=dV: adir(dV, U); ttl = rf"Ng=0, $\Delta V={dV:g}$"
    if "A" in what: mav.make_video(get, ttl, f"{d}/A_omega_vs_U.mp4", ed_fn=ed_A)
    if "weights" in what: mw.make_video(get, ed_p, ttl, f"{d}/qp_weights_vs_U.mp4")
    if "orbitals" in what or "density" in what:
        fr = mo.load_frames(get)
        if "orbitals" in what: mo.make_orbitals(fr, ttl, f"{d}/qp_orbitals_vs_U.mp4")
        if "density" in what: mo.make_density(fr, ttl, f"{d}/qp_density_vs_U.mp4")
