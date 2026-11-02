"""Ng=2: track the gGA solution of the half-filled dimer from dV=0 to dV=5 (step 0.2) on the U grid 0.1..8+2*dV (step 0.1), direct minimization (not_imp_gga_dimer).
Stage 1 (per U, parallel): continuation in dV from the saved dV=0 solutions (scan_ng2_not_imp lowest-energy branch, extended above U=10 by continuation in U).
Stage 2 (per dV, parallel): sweeps in U (up, down, up), each point also seeded from its U-neighbour; a point is replaced only if the energy is lower.
EVERY converged point is saved at once: scan_Uc_vs_dV/dV_<dV>/U_<U>/{z.npy, output.out}, and one log line per point goes to scan_Uc_vs_dV/log.txt. Resumable (stage 1 skips points on disk).
usage (repo root): python3 scripts/scan_Uc_vs_dV.py [NPROC]"""
import sys, os, re, numpy as np
from multiprocessing import Pool
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
sys.path.insert(0, "scripts")
import scan_ng2_not_imp_dV2_U as sc
import not_imp_gga_dimer as ni
t = 1.0; DVS = [round(0.2 * k, 1) for k in range(26)]; umax = lambda dV: round(8 + 2 * dV, 1)
UGRID = [round(0.1 * k, 1) for k in range(1, 201)]; ROOT = "scan_Uc_vs_dV"
pdir = lambda dV, U: f"{ROOT}/dV_{dV:.1f}/U_{U:.1f}"

def solve(U, dV, z0):
    m = ni.NotImpDimer(2, U, t, deltaV=dV); z, npass, viol, tg = sc.converge(m, z0)
    return m.energy(z, grad=False), z, viol, tg

def save(U, dV, b, tag):
    d = pdir(dV, U); os.makedirs(d, exist_ok=True); E, z, viol, tg = b
    np.save(d + "/z.npy", z)
    line = f"U/t={U:.1f} dV/t={dV:.1f} E_var/t={E:.8f} |c|={viol:.1e} tg={tg:.1e} conv={viol < 1e-9 and tg < 5e-6} stage={tag}"
    open(d + "/output.out", "w").write(line + "\n")
    with open(f"{ROOT}/log.txt", "a") as f: f.write(line + "\n")

def load(U, dV):
    d = pdir(dV, U)
    if not os.path.exists(d + "/output.out"): return None
    s = open(d + "/output.out").read(); g = lambda k: float(re.search(k + r"=([\d.eE+-]+)", s).group(1))
    return g("E_var/t"), np.load(d + "/z.npy"), g(r"\|c\|"), g("tg")

def start_dV0():
    lo = {}
    for line in open("scan_ng2_not_imp/lowest_energy.txt"):
        if not line.startswith("#") and line.strip(): f = line.split(); lo[round(float(f[0]), 1)] = f[2]
    z = {U: np.load(f"scan_ng2_not_imp/{lo[U]}/U_{U:.1f}/z.npy") for U in lo}; zp = z[10.0]
    for U in UGRID:
        if U > 10.0: zp = solve(U, 0.0, zp)[1]; z[U] = zp
    return z

def chain(args):
    U, z = args
    for dV in DVS:
        if U > umax(dV): continue
        b = load(U, dV)
        if b is None: b = solve(U, dV, z); save(U, dV, b, "chain")
        z = b[1]
    return U

def sweep(args):
    dV, order, r = args; n = 0
    for k in range(1, len(order)):
        U, Un = order[k], order[k - 1]; cur, nb = load(U, dV), load(Un, dV)
        if cur is None or nb is None: continue
        b = solve(U, dV, nb[1])
        if b[2] < 1e-8 and b[0] < cur[0] - 1e-9: save(U, dV, b, f"sweep{r}"); n += 1
    return dV, n

if __name__ == "__main__":
    npr = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    os.makedirs(ROOT, exist_ok=True); z0 = start_dV0(); print("dV=0 start solutions:", len(z0), flush=True)
    with Pool(npr) as p:
        for i, U in enumerate(p.imap_unordered(chain, [(U, z0[U]) for U in UGRID])): pass
        print("stage 1 (dV continuation) done", flush=True)
        for r in range(3):
            order = lambda dV: [U for U in (UGRID if r != 1 else UGRID[::-1]) if U <= umax(dV)]
            out = p.map(sweep, [(dV, order(dV), r) for dV in DVS]); print(f"sweep {r}: improved points per dV:", {dV: n for dV, n in out}, flush=True)
