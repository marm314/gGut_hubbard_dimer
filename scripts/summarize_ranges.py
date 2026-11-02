"""Summary of the sampled allowed intervals of the ghost poles (cederbaum_data/scan_ghost_tc_eps1e-07/range_V*.npz): all valid shifts and with aufbau. /usr/bin/python3 scripts/summarize_ranges.py"""
import numpy as np
for V in (0.7071067811865476, 1.4142135623730951):
    r = np.load(f"cederbaum_data/scan_ghost_tc_eps1e-07/range_V{V}.npz")
    for B in (3, 5):
        for q in range(1 if B == 3 else 2):
            out = []
            for key in ("", "_auf"):
                ks = [k for k in r.files if k.endswith(f"B{B}_lo{key}")]; lo = np.array([r[k][q] for k in ks]); hi = np.array([r[k.replace("_lo", "_hi")][q] for k in ks]); out.append((lo, hi))
            (la, ha), (lb, hb) = out
            print(f"V={V:.3f} B_c={B} ghost {q+1}: n={len(la)}; all valid: lower median {np.median(la):.0f} (min {la.min():.0f}, max {la.max():.0f}), upper median {np.median(ha):.0f} (min {ha.min():.1f}, max {ha.max():.0f}); "
                  f"aufbau: lower median {np.median(lb):.0f} (min {lb.min():.0f}, max {lb.max():.0f}), ceiling median {np.median(hb):.2f} (min {hb.min():.2f}, max {hb.max():.2f})")
