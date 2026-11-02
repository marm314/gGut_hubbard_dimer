"""Independent re-check of every placed point (rebuilds H_qp from the stored gauge parameters via gauge_tools): eigenvalues of rho (0/1), occupations n_k=phi_k^T rho phi_k of ALL H_qp levels (0/1),
aufbau (highest rho-occupied level below lowest rho-empty level), commutator before/after the shift, QP pole/weight unchanged, number of occupied levels. PYTHONPATH=scripts python3 scripts/verify_placed.py"""
import numpy as np, re
import gauge_tools as gt
for eps, d_dir, tcs in ((1e-7, "ghost_placed", [round(0.2 * i, 2) for i in range(1, 26)]), (1e-9, "ghost_placed_eps1e-09", [0.05, 0.1])):
    for V in (0.7071067811865476, 1.4142135623730951):
        for B in (3, 5):
            w = dict(rho=0, occ=0, comm=0, dqp=0, dw=0, n_aufbau=0, n=0, nocc_ok=True)
            for tc in tcs:
                r1 = gt.solution_at_gauge(V, tc, B, placed=True, eps=eps); r0 = gt.solution_at_gauge(V, tc, B, placed=False, eps=eps)
                lam = np.linalg.eigvalsh(r1["rho"]); w["rho"] = max(w["rho"], np.minimum(abs(lam), abs(1 - lam)).max()); n = r1["occ"]; w["occ"] = max(w["occ"], np.minimum(abs(n), abs(1 - n)).max())
                e = r1["eps"]; occd = e[n > 0.5]; empt = e[n <= 0.5]; w["n_aufbau"] += int(occd.max() < empt.min()); w["n"] += 1; w["nocc_ok"] &= (len(occd) == r1["Nocc"])
                w["comm"] = max(w["comm"], abs(r1["comm"] - r0["comm"])); k1, k0 = np.argmax(r1["W"][0]), np.argmax(r0["W"][0]); w["dqp"] = max(w["dqp"], abs(r1["eps"][k1] - r0["eps"][k0])); w["dw"] = max(w["dw"], abs(r1["W"][0, k1] - r0["W"][0, k0]))
            print(f"eps={eps:g} V/g={V:.3f} B_c={B} ({w['n']} points): rho eigenvalues within {w['rho']:.1e} of 0/1; occupations of ALL H_qp levels within {w['occ']:.1e} of 0/1; occupied levels = N_occ at all points: {w['nocc_ok']}; "
                  f"aufbau (all occupied below all empty) at {w['n_aufbau']}/{w['n']}; |change of commutator| {w['comm']:.1e}; |change of QP pole| {w['dqp']:.1e}; |change of QP weight| {w['dw']:.1e}", flush=True)
