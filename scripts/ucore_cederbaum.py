"""Core Hubbard U for the t_c-Cederbaum model: H -> H + U n_c,up n_c,dn, in the exact diagonalization (exact_cederbaum.solve, patched here)
and in the direct-minimization gGut solver (not_imp_gga_cederbaum.Problem; U added to the core fragment's H_loc, which enters only through
<phi_c|H_loc|phi_c> and its gradient, so no new gradient terms). Separate module: the existing files are not modified.
Functions: ed_U(V,tc,U), model_U(V,tc,Bc,U,eps), and a test/scan driver (see __main__)."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import exact_cederbaum as ed
import gga_cederbaum as g
import not_imp_gga_cederbaum as N

_orig_build = ed.build_hamiltonian


def ed_U(V, tc, U, Delta=1.0, Ec=10.0, ht="cv"):
    """Exact solution with core U (spectrum of G^h_cc, E0, nu_c)."""
    def build(Delta_, Ec_, V_, tc_=0.0, ht_="co"):
        H, c, n = _orig_build(Delta_, Ec_, V_, tc_, ht_)
        return H + U * (n[ed.C_UP] @ n[ed.C_DN]), c, n
    ed.build_hamiltonian = build
    try:
        return ed.solve(Delta, Ec, V, tc, ht=ht)
    finally:
        ed.build_hamiltonian = _orig_build


def model_U(V, tc, Bc, U, eps, Delta=1.0, Ec=10.0, hop="cv"):
    """gGut model with U n_c,up n_c,dn added to the core fragment H_loc (dense, projected on the half-filled sector)."""
    m = g.Model(Delta, Ec, V, tc, Bc, eps, hop)
    f = m.frag[0]; ops = f.ops
    f.Hloc = f.Hloc + U * (ops["n_pu"] @ ops["n_pd"])[f.sec][:, f.sec].toarray()
    return m


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ed":                      # python3 ucore_cederbaum.py ed V tc : nu_c and spectrum vs U
        V, tc = float(sys.argv[2]), float(sys.argv[3])
        for U in [0, 2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 20]:
            r = ed_U(V, tc, U); s = r["weights"] > 1e-3
            print(f"U={U:5.1f} E0={r['E0']:.6f} nu_c={r['nu_c']:.5f} ndeg={r['ndeg']} poles {np.round(r['poles'][s],3)} w {np.round(r['weights'][s],3)}", flush=True)
    elif mode == "check":                 # python3 ucore_cederbaum.py check V tc Bc U eps : U=0 reproduces old Hloc; gradients; B_c=1 energy
        V, tc, Bc, U, eps = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
        m0 = g.Model(1.0, 10.0, V, tc, Bc, eps, "cv"); m1 = model_U(V, tc, Bc, 0.0, eps)
        print("U=0 Hloc identical to the original:", np.abs(m0.frag[0].Hloc - m1.frag[0].Hloc).max())
        m = model_U(V, tc, Bc, U, eps); d = m.frag[0].Hloc - m0.frag[0].Hloc
        print(f"U={U}: Hloc change has diag in [{np.diag(d).min():.1f},{np.diag(d).max():.1f}], symmetric {np.abs(d-d.T).max():.1e}")
        P = N.Problem(m); N.check_gradients(P, np.random.default_rng(1))
    elif mode == "scan":                  # python3 ucore_cederbaum.py scan V tc Bc eps U1,U2,... : gGut vs ED for each U (continuation in U, random-start fallback)
        V, tc, Bc, eps = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]); Us = [float(x) for x in sys.argv[6].split(",")]
        out = f"../cederbaum_data/ucore/V{V:.4f}_tc{tc}_B{Bc}_eps{eps:g}"; os.makedirs(out, exist_ok=True)
        zprev = None
        if Bc == 3 and Us[0] == 0.0:
            zprev = np.load(f"../cederbaum_data/scan_ghost_tc_eps{eps:g}/V{V}.npz")[f"tc{tc}_B3_z"]
        for U in Us:
            t0 = time.time(); r = ed_U(V, tc, U); sat = r["weights"] > 1e-3
            m = model_U(V, tc, Bc, U, eps); P = N.Problem(m); rng = np.random.default_rng(int(10 * U) + 7); best = None
            for i, z0 in enumerate(([zprev] if zprev is not None else []) + [None] * 30):
                res = N.minimize_once(P, z0 if z0 is not None else N.random_start(P, rng), maxiter=500)
                cons = np.abs(P.constraints(res.x)).max()
                if cons < 1e-9:
                    o = N.recover(P, res.x)
                    if o["comm"] < 1e-4 and (best is None or res.fun < best[0].fun): best = (res, o)
                if best is not None and best[0].fun - r["E0"] < 1e-3: break
            print(f"U={U}: ED E0={r['E0']:.6f} nu_c={r['nu_c']:.5f} (ndeg {r['ndeg']}) poles {np.round(r['poles'][sat],3)} w {np.round(r['weights'][sat],3)}", flush=True)
            if best is None:
                print(f"    gGut: no accepted solution ({time.time()-t0:.0f}s)", flush=True); continue
            res, o = best; zprev = res.x
            print(f"    gGut B_c={Bc}: E-E_ED={res.fun-r['E0']:.3e} nu_c={o['nu']:.5f} kkt={o['kkt']:.1e} comm={o['comm']:.1e} poles {np.round(o['poles'],3)} w {np.array2string(o['weights'],precision=4)} ({time.time()-t0:.0f}s)", flush=True)
            np.savez(f"{out}/U_{U:g}.npz", z=res.x, E=res.fun, E_ED=r["E0"], nu=o["nu"], poles=o["poles"], w=o["weights"], ed_poles=r["poles"], ed_w=r["weights"], ed_nu=r["nu_c"])
