"""Check that rho is a clean 0/1 occupation at every converged point and at shifted multipliers (gauge moves).
(1) eigenvalues of rho (distance to 0 or 1); (2) occupations n_k = phi_k^T rho phi_k of the H_qp eigenlevels at mu0 + t * null-direction for t in +-(0,5,20,100,300), axes + random rays.
python3 scripts/check_ghost_occ.py"""
import sys
src = open("scripts/ghost_range_tc.py").read().split("for tc in [round")[0]
sys.argv = ["x", "0.7071067811865476"]
src = src.replace('return e, (occ', 'LAST["occ"] = occ; LAST["rho"] = o["rho"]; return e, (occ'); LAST = {}
exec(src)
import numpy as np
for V in (0.7071067811865476, 1.4142135623730951):
    data = dict(np.load(f"cederbaum_data/scan_ghost_tc/V{V}.npz")); worst_rho = 0; worst_n = 0; worst_n_valid = 0; nshift = 0; nacc = 0; rej_comm = []; rej_dev = []
    for tc in [round(0.2 * i, 2) for i in range(1, 26)]:
        for Bc in (3, 5):
            mu0, null, analyse, rank = setup(Bc, tc); res0 = analyse(mu0); ng0 = int(res0[1].sum()); c0 = res0[2]; qp0 = res0[3]; wq0 = res0[5]; wt0 = res0[6]
            lam = np.linalg.eigvalsh(LAST["rho"]); worst_rho = max(worst_rho, np.minimum(abs(lam), abs(1 - lam)).max())
            dirs = list(np.eye(null.shape[1])) + list(np.random.default_rng(2).standard_normal((5, null.shape[1])))
            for d in dirs:
                d = d / np.linalg.norm(d)
                for t in (-300, -100, -20, -5, 0, 5, 20, 100, 300):
                    r = analyse(mu0 + t * (null @ d)); n = LAST["occ"]; dev = np.minimum(abs(n), abs(1 - n)).max(); nshift += 1
                    ok = r[1].sum() == ng0 and r[2] <= c0 + 1e-6 and abs(r[3] - qp0) < 1e-4 and abs(r[5] - wq0) < 1e-5 and abs(r[6] - wt0) < 1e-5
                    if ok: nacc += 1; worst_n_valid = max(worst_n_valid, dev)
                    else: rej_comm.append(r[2]); rej_dev.append(dev)
    print(f"V={V}: {nshift} shifted H_qp checked; worst |rho eigenvalue - nearest 0/1| = {worst_rho:.1e}; ACCEPTED shifts: {nacc}, worst |n_k - nearest 0/1| = {worst_n_valid:.1e}; REJECTED shifts: {nshift-nacc}, their worst commutator {max(rej_comm):.1e}, worst |n_k-0/1| {max(rej_dev):.2f}", flush=True)
