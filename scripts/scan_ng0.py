"""Ng=0 (ordinary Gutzwiller) half-filled dimer: U-continuation scans of the direct minimization (NotImpDimer(0,...)) for dV = 0..5 (step 0.2) and 10,
U = 0.1..15+3*dV (step 0.1). Forward (from U=0.1, 20 random starts there) and backward (from UMAX, 20 random starts) branches; the lowest energy at each U is kept in lowest_energy.txt.
Every converged point is saved at once: scan_ng0/dV_<dV>/{forward,backward}/U_<U>/{z.npy,output.out}; one log line per point in scan_ng0/log.txt. usage (repo root): python3 scripts/scan_ng0.py [NPROC]"""
import sys, os, numpy as np
from multiprocessing import Pool
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
sys.path.insert(0, "scripts")
import not_imp_gga_dimer as ni
import scan_ng2_not_imp_dV2_U as sc
t = 1.0; ROOT = "scan_ng0"; DVS = [round(0.2 * k, 1) for k in range(26)] + [10.0]
def grid(dV): return [round(0.1 * k, 1) for k in range(1, int(round((15 + 3 * dV) * 10)) + 1)]
def point(dV, U, z0, name):
    m = ni.NotImpDimer(0, U, t, deltaV=dV)
    if z0 is None:
        rng = np.random.default_rng(int(U * 10)); best = None
        for _ in range(20):
            r = m.minimize(m.random_start(rng))
            if np.abs(m.constraints(r.x)).max() < 1e-8 and (best is None or r.fun < best.fun - 1e-12): best = r
        z0 = best.x
    z, _, viol, tg = sc.converge(m, z0); a = m.analyse(z); E = m.energy(z, grad=False)
    d = f"{ROOT}/dV_{dV:.1f}/{name}/U_{U:.1f}"; os.makedirs(d, exist_ok=True); np.save(d + "/z.npy", z)
    line = f"U/t={U:.1f} dV/t={dV:.1f} E_var/t={E:.8f} Z0={a['Z0']:.6f} Z1={a['Z1']:.6f} |c|={viol:.1e} tg={tg:.1e} conv={viol < 1e-9 and tg < 5e-6} branch={name}"
    open(d + "/output.out", "w").write(line + "\n")
    with open(f"{ROOT}/log.txt", "a") as f: f.write(line + "\n")
    return E, z
def run(dV):
    Us = grid(dV); res = {}
    for name, order in (("forward", Us), ("backward", Us[::-1])):
        z = None
        for U in order:
            E, z = point(dV, U, z, name); res[(name, U)] = E
    with open(f"{ROOT}/dV_{dV:.1f}/lowest_energy.txt", "w") as f:
        f.write("# U/t  E_var/t  branch\n")
        for U in Us:
            ef, eb = res[("forward", U)], res[("backward", U)]; f.write(f"{U:5.1f} {min(ef, eb):.8f} {'forward' if ef <= eb else 'backward'}\n")
    return dV
if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as p:
        for dV in p.imap_unordered(run, DVS): print("done dV =", dV, flush=True)
