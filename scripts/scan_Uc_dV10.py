"""Ng=2: U_c at dV=10 (not added to the dV<=5 plot). Starting from the saved dV=5 solutions of scan_Uc_vs_dV/ (U<=18, extended here to U=UMAX by forward continuation in U),
continue in dV (5.2,...,10, step 0.2) at every U on 10.0..UMAX (step 0.1), then two U-sweeps at dV=10. Every converged point is saved at once:
scan_Uc_dV10/dV_<dV>/U_<U>/{z.npy,output.out}, one log line per point in scan_Uc_dV10/log.txt. usage (repo root): python3 scripts/scan_Uc_dV10.py [NPROC] [UMAX]"""
import sys, os, re, numpy as np
from multiprocessing import Pool
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
sys.path.insert(0, "scripts")
import scan_ng2_not_imp_dV2_U as sc
import not_imp_gga_dimer as ni
t = 1.0; ROOT = "scan_Uc_dV10"; DVS = [round(5.2 + 0.2 * k, 1) for k in range(26)]
UMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
UGRID = [round(0.1 * k, 1) for k in range(100, int(round(UMAX * 10)) + 1)]
pdir = lambda dV, U: f"{ROOT}/dV_{dV:.1f}/U_{U:.1f}"

def solve(U, dV, z0):
    m = ni.NotImpDimer(2, U, t, deltaV=dV); z, _, viol, tg = sc.converge(m, z0); return m.energy(z, grad=False), z, viol, tg

def save(U, dV, b, tag):
    d = pdir(dV, U); os.makedirs(d, exist_ok=True); E, z, viol, tg = b; np.save(d + "/z.npy", z)
    line = f"U/t={U:.1f} dV/t={dV:.1f} E_var/t={E:.8f} |c|={viol:.1e} tg={tg:.1e} conv={viol < 1e-9 and tg < 5e-6} stage={tag}"
    open(d + "/output.out", "w").write(line + "\n")
    with open(f"{ROOT}/log.txt", "a") as f: f.write(line + "\n")

def load(U, dV):
    d = pdir(dV, U)
    if not os.path.exists(d + "/output.out"): return None
    s = open(d + "/output.out").read(); g = lambda k: float(re.search(k + r"=([\d.eE+-]+)", s).group(1))
    return g("E_var/t"), np.load(d + "/z.npy"), g(r"\|c\|"), g("tg")

def start_dV5():
    z, zp = {}, None
    for U in UGRID:
        f = f"scan_Uc_vs_dV/dV_5.0/U_{U:.1f}/z.npy"
        if os.path.exists(f): zp = np.load(f)
        else: b = solve(U, 5.0, zp); zp = b[1]; save(U, 5.0, b, "dV5-extension")
        z[U] = zp
    return z

def chain(args):
    U, z = args
    for dV in DVS:
        b = load(U, dV)
        if b is None: b = solve(U, dV, z); save(U, dV, b, "chain")
        z = b[1]
    return U

def sweep(args):
    order, r = args; n = 0
    for k in range(1, len(order)):
        U, Un = order[k], order[k - 1]; cur, nb = load(U, 10.0), load(Un, 10.0)
        b = solve(U, 10.0, nb[1])
        if b[2] < 1e-8 and b[0] < cur[0] - 1e-9: save(U, 10.0, b, f"sweep{r}"); n += 1
    return n

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True); z0 = start_dV5(); print("dV=5 start solutions:", len(z0), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as p:
        for _ in p.imap_unordered(chain, [(U, z0[U]) for U in UGRID]): pass
        print("chain done", flush=True)
    for r, order in enumerate((UGRID, UGRID[::-1])): print(f"sweep {r}: improved", sweep((order, r)), flush=True)
    print("all done", flush=True)
