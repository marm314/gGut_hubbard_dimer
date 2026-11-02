"""Refine the dV=2 confirmed points by direct minimization seeded at the confirmed (R,lambda).
Saves z.npy, R.dat, lambda.dat and a one-line output.out per point in scan_ng{Ng}_dV2_refined/U_x/ and logs every point.
usage (repo root): python3 scripts/refine_dV2.py NG [Umin Umax]"""
import sys, os, time, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni, make_qp_orbital_videos as q
Ng = int(sys.argv[1]); lo, hi = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.1, 10.0)
for U in np.round(np.arange(lo, hi + 1e-9, 0.1), 1):
    t0 = time.time(); src = f"scan_ng{Ng}_dV2_confirmed/U_{U:.1f}"; dst = f"scan_ng{Ng}_dV2_refined/U_{U:.1f}"
    R, lam = q.load_R_lambda(src); m = ni.NotImpDimer(Ng, float(U), deltaV=2.0)
    res = m.minimize(m.start_from_R_lambda(R, lam)); a = m.analyse(res.x)
    os.makedirs(dst, exist_ok=True); np.save(f"{dst}/z.npy", res.x)
    np.savetxt(f"{dst}/refined_R.dat", np.array([a["R0"], a["R1"]])); np.savetxt(f"{dst}/refined_lambda.dat", np.vstack([a["lam0"], a["lam1"]]))
    pphi = sum((x @ m.Ndd @ x) / (x @ x) for x in m.unpack(res.x)[:2])
    line = f"U/t={U:.1f} E_var/t={a['E_var']:.8f} Pion_phi={pphi:.8f} docc={a['docc']:.8f} constraint={a['constraint']:.1e} kkt_imp={a['kkt_imp']:.1e} success={res.success}"
    open(f"{dst}/output.out", "w").write(line + "\n"); print(line + f" ({time.time()-t0:.1f}s)", flush=True)
