"""||[H_qp,rho]|| at the min-norm gauge for every converged point of the eps=1e-9 scan (the solutions are not stationary in X). PYTHONPATH=scripts python3 scripts/comm_scan.py"""
import sys
src = open("scripts/ghost_range_tc.py").read().split("for tc in [round")[0]; sys.argv = ["x", "0.7071067811865476"]; exec(src)
import numpy as np
for V in (0.7071067811865476, 1.4142135623730951):
    data = dict(np.load(f"cederbaum_data/scan_ghost_tc/V{V}.npz")); bad = {3: [], 5: []}
    for tc in [round(0.2 * i, 2) for i in range(1, 26)]:
        for Bc in (3, 5):
            mu0, null, analyse, rank = setup(Bc, tc); c = analyse(mu0)[2]
            if c > 1e-6: bad[Bc].append((tc, float(f"{c:.1e}")))
    print(V, "points with ||[H_qp,rho]|| > 1e-6:", bad)
