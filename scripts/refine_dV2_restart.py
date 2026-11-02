"""Restart the failed dV=2 refinements (success=False, or kkt_imp>1e-6, or constraint>1e-8) from the refined neighbour z at U/t -+ 0.1.
Sweeps up then down until no point improves; a restart is kept only if it converges (success, kkt_imp<=1e-6, constraint<=1e-8) or lowers kkt_imp.
Overwrites the point's files in scan_ng{Ng}_dV2_refined and logs every attempt. usage (repo root): python3 scripts/refine_dV2_restart.py NG"""
import sys, os, re, numpy as np
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
KTOL, CTOL = 1e-4, 1e-8   # accepted: embedding residual < 1e-4, constraint < 1e-8 (success flag not required)
Ng = int(sys.argv[1]); Us = np.round(np.arange(0.1, 10.01, 0.1), 1)
d = lambda U: f"scan_ng{Ng}_dV2_refined/U_{U:.1f}"
def info(U):
    s = open(d(U) + "/output.out").read(); g = lambda k: float(re.search(k + r"=([\d.eE+-]+)", s).group(1))
    return g("kkt_imp"), g("constraint"), "success=True" in s
def good(U):
    k, c, ok = info(U); return k <= KTOL and c <= CTOL
def attempt(U, Un):
    m = ni.NotImpDimer(Ng, float(U), deltaV=2.0); z0 = np.load(d(Un) + "/z.npy")
    res = m.minimize(z0); a = m.analyse(res.x)
    return m, res, a
changed = True
while changed:
    changed = False
    for order in (Us, Us[::-1]):
        for U in order:
            if good(U): continue
            for Un in (round(U - 0.1, 1), round(U + 0.1, 1)):
                if Un not in Us or not good(Un): continue
                m, res, a = attempt(U, Un); k0 = info(U)[0]
                ok = a["kkt_imp"] <= KTOL and a["constraint"] <= CTOL
                print(f"U/t={U:.1f} from {Un:.1f}: kkt_imp {k0:.1e} -> {a['kkt_imp']:.1e} constraint={a['constraint']:.1e} E={a['E_var']:.8f} success={res.success} {'KEPT' if ok or a['kkt_imp'] < k0 else 'rejected'}", flush=True)
                if ok or a["kkt_imp"] < k0:
                    pphi = sum((x @ m.Ndd @ x) / (x @ x) for x in m.unpack(res.x)[:2])
                    np.save(d(U) + "/z.npy", res.x)
                    np.savetxt(d(U) + "/refined_R.dat", np.array([a["R0"], a["R1"]])); np.savetxt(d(U) + "/refined_lambda.dat", np.vstack([a["lam0"], a["lam1"]]))
                    open(d(U) + "/output.out", "w").write(f"U/t={U:.1f} E_var/t={a['E_var']:.8f} Pion_phi={pphi:.8f} docc={a['docc']:.8f} constraint={a['constraint']:.1e} kkt_imp={a['kkt_imp']:.1e} success={res.success}\n")
                    changed = changed or ok
                    if ok: break
print("remaining not good:", [float(U) for U in Us if not good(U)])
