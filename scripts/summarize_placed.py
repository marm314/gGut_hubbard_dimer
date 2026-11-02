"""Summary statistics of cederbaum_data/ghost_placed/V*.npz (used for the numbers in doc/cederbaum_gauge.tex). /usr/bin/python3 scripts/summarize_placed.py"""
import numpy as np
for V in (0.7071067811865476, 1.4142135623730951):
    d = np.load(f"cederbaum_data/ghost_placed/V{V}.npz")
    for B in (3, 5):
        ks = [k for k in d.files if k.endswith(f"B{B}_targets")]; P = lambda s: [k[:-len("targets")] + s for k in ks]; g = lambda s: np.array([d[x] for x in P(s)])
        maxres = max(np.abs(np.array(d[k[:-7] + "ghost_after"]) - np.array(d[k])).max() for k in ks)
        print(f"V={V:.3f} B_c={B}: n={len(ks)}; max|ghost-target| {maxres:.1e}; max lin.res {g('lin_res').max():.1e}; max|dQP| {np.abs(g('qp_after')-g('qp_before')).max():.1e}; max|dwQP| {np.abs(g('wq_after')-g('wq_before')).max():.1e}; "
              f"max|dw_occ| {np.abs(g('wocc_after')-g('wocc_before')).max():.1e}; max occdev {g('occdev_after').max():.1e}; max comm {g('comm_after').max():.1e}; aufbau true {int(g('aufbau_after').sum())}/{len(ks)}; "
              f"QP overlap min {g('qp_overlap').min():.6f}; ghostproj max {np.nanmax(g('ghostproj_diff')):.1e}; max ghost core weight {np.nanmax(np.array([d[x] for x in P('ghostw_after')])):.1e}; lifted empty ghosts at {int((g('lifted')>0).sum())} points")
        rb, ra = g("rankPhi_before"), g("rankPhi_after"); print(f"    rank(Phi_c): before min {rb.min()} median {int(np.median(rb))} max {rb.max()}; after min {ra.min()} median {int(np.median(ra))} max {ra.max()} (H_imp,c dim {36 if B == 3 else 400})")
    if B == 5:
        ea, et = g("empty_after"), g("empty_target"); print(f"    free empty ghost (B_c=5): max|level-target| {np.nanmax(np.abs(ea-et)):.1e}; targets {np.nanmin(et):.2f}..{np.nanmax(et):.2f}; before {np.nanmin(g('empty_before')):.2f}..{np.nanmax(g('empty_before')):.2f}; max ED target weight {np.nanmax(g('empty_target_w')):.1e}, min {np.nanmin(g('empty_target_w')):.1e}")
