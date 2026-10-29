"""Measure peak positions in the rendered Fig. 2b,c (G0W0 green, exact red, GW+C blue) of Kocklauner & Golze, arXiv:2412.17132 (page 7 of the PDF
in GhostSatellites_Cederbaum/). Positions relative to the QP peak of each curve, in units of Omega, where Omega = mean spacing of the GW+C (blue)
satellites in panel c (rho/Omega = 1.0) [panel b: the single blue spacing is printed as a cross-check]. Accuracy: a few per cent (pixel-level).
Saves cederbaum_data/golze_fig2_measured.npz and prints everything. python3 measure_golze_fig2.py"""
import subprocess, os
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

PDF = "/home/marm314/C_MEJUTO/gGut_notes/gGut/gGut_on_models/GhostSatellites_Cederbaum/Kocklauner_Golze_2412.17132.pdf"
tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "golze_fig2")
subprocess.run(["pdftoppm", "-f", "7", "-l", "7", "-r", "400", "-png", PDF, tmp], check=True)
im = np.array(Image.open(tmp + "-07.png").convert("RGB")).astype(int); H, W, _ = im.shape
y0, y1 = int(0.04 * H), int(0.2 * H)
cols = {"blue": (31, 119, 180), "green": (44, 160, 44), "red": (214, 39, 40)}
res = {}
for pname, (a, b) in {"b": (0.37, 0.63), "c": (0.64, 0.91)}.items():
    sub = im[y0:y1, int(a * W):int(b * W)]
    for n, c in cols.items():
        h = (np.abs(sub - np.array(c)).sum(2) < 60).sum(0).astype(float)
        pk, _ = find_peaks(h, height=0.15 * h.max(), distance=15)
        res[(pname, n)] = (pk.astype(float), h[pk])
        print(f"panel {pname} {n}: peak x (px) {pk.tolist()} heights {h[pk].astype(int).tolist()}")
bc = res[("c", "blue")][0]; Om_c = np.mean(np.diff(np.sort(bc)))
bb = np.sort(res[("b", "blue")][0]); Om_b = np.diff(bb)[0]
print(f"Omega (px): panel c mean blue spacing {Om_c:.1f} (spacings {np.diff(np.sort(bc)).tolist()}); panel b blue spacing {Om_b:.1f}")
out = {}
for pname, rho in (("b", 0.5), ("c", 1.0)):
    for n, lab in (("green", "g0w0"), ("red", "exact")):
        pk = np.sort(res[(pname, n)][0])[::-1]            # right to left: QP first
        rel = (pk - pk[0]) / Om_c
        out[f"rho{rho}_{lab}_rel"] = rel
        print(f"rho/Omega={rho} {lab}: positions relative to QP (Omega) {np.round(rel,3)}  [using panel-b spacing: {np.round((pk-pk[0])/Om_b,3)}]")
np.savez("cederbaum_data/golze_fig2_measured.npz", **out, Omega_px_c=Om_c, Omega_px_b=Om_b)

# full digitized G0W0 (green) curve: topmost green pixel of each column, intensity = baseline - row (px); x as above (QP green peak = 0, units of Omega)
for pname, rho in (("b", 0.5), ("c", 1.0)):
    a, b = {"b": (0.37, 0.63), "c": (0.64, 0.91)}[pname]
    yy0, yy1 = int(0.04 * H), int(0.2 * H)
    sub = im[yy0:yy1, int(a * W):int(b * W)]
    m = np.abs(sub - np.array(cols["green"])).sum(2) < 60
    if pname == "b":
        m[:int(0.0525 * H), :int(0.1 * W)] = False             # legend line of panel b
    ok = m.any(0)
    top = np.where(ok, m.argmax(0), -1)
    base = np.median(top[ok])                                  # flat part of the curve
    xs = np.arange(m.shape[1])[ok]
    qp = np.sort(res[(pname, "green")][0])[::-1][0]
    out[f"rho{rho}_g0w0_curve_x"] = (xs - qp) / Om_c
    out[f"rho{rho}_g0w0_curve_y"] = base - top[ok]
    mr = np.abs(sub - np.array(cols["red"])).sum(2) < 60
    if pname == "b":
        mr[:int(0.0525 * H), :int(0.1 * W)] = False             # legend line
    okr = mr.any(0); topr = np.where(okr, mr.argmax(0), -1)
    out[f"rho{rho}_exact_hpx"] = np.median(topr[okr]) - topr[okr].min()      # height of the (unclipped) exact QP peak in px: calibrates the intensity scale (same y scale for all curves of a panel)
    print(f"rho/Omega={rho}: green curve {ok.sum()} columns, baseline row {base}, max height {np.max(base - top[ok])} px")
np.savez("cederbaum_data/golze_fig2_measured.npz", **out, Omega_px_c=Om_c, Omega_px_b=Om_b)
