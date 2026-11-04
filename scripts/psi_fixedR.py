"""Self-consistent gGA solution at a FIXED R: minimize the ghost-GA functional (not_imp_gga_dimer.NotImpDimer, SLSQP) with the extra equality
constraints R_i(phi_i) = R'_i (R'_i = modified R of the solution in the gauge of its z.npy), warm-started along s from the solution (s = 1)
up and down. Each converged point is saved at once (../Rfixed/ng<Ng>_<tag>_U<U>_<part>/s_<s>/z.npy) and appended to
../psi_scan/Rfix_ng<Ng>_<tag>_U<U>_<part>.txt: s, E_var(gGA functional), constraint violation, |R'|, P_ion(Psi), E_Psi, overlap with FCI, R_Phi - R' of Psi.
part "norm": fix only |R_i|^2 -> (s |R_i|)^2 (R_i itself is only defined up to the orthogonal rotations of the qp basis of the site, a symmetry of the
functional: fixing the vector with |R_i| unchanged ("rot") gives back the same energy and the same Psi, as found).
Continuation in s from the unmodified solution (s=1), up and down, each point started from the previous converged one; a point is accepted if SLSQP
reports success and the constraint violation is < 1e-7, otherwise the step in s is halved (from the last accepted point) down to 1e-3 and the walk
stops there. ds is the largest step. Usage: python psi_fixedR.py <dV0|dV2|dV2n> <Ng> <U> <none|rot|norm> [s_min s_max ds]"""
import os
import sys
import time
import numpy as np
import gutzwiller_psi as G
from not_imp_gga_dimer import NotImpDimer

tag, ng, U, part = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
smin, smax, ds = (float(x) for x in sys.argv[5:8]) if len(sys.argv) >= 8 else (0.7, 1.3, 0.05)
d, dV = G.solution_dir(tag, ng, U)
model = NotImpDimer(ng, U, deltaV=dV)
N = model.N
z_sol = np.load(os.path.join(d, "z.npy")); assert len(z_sol) == model.nz
base = f"../Rfixed/ng{ng}_{tag}_U{U:.1f}_{part}"; os.makedirs(base, exist_ok=True)
txt = f"../psi_scan/Rfix_ng{ng}_{tag}_U{U:.1f}_{part}.txt"
open(txt, "w").write("# s  E_var  constraint  |R'_0|  P_ion  E_Psi  overlap  |R_Phi-R'|_0   (Psi from the fixed-R solution)\n")


def Rprime(s):
    if part == "norm":
        x0, x1, _ = model.unpack(z_sol)                                  # |R_i|^2 of the solution from phi_i (no R files for Ng=0)
        return [np.array([s * s * (model.frag(x, i)["R"] @ model.frag(x, i)["R"])]) for i, x in enumerate((x0, x1))]    # target |R_i|^2 (1-vectors)
    return [np.array(x) for x in G.build_psi(d, U, dV, Ng=ng, verbose=False, Rscale=(s, s), mode="B", part=None if part == "none" else part)["R"]]


def Rfun(z):
    x0, x1, _ = model.unpack(z)
    Rs = [model.frag(x0, 0)["R"], model.frag(x1, 1)["R"]]
    return np.array([r @ r for r in Rs]) if part == "norm" else np.concatenate(Rs)


def Rjac(z, h=1e-6):
    J = np.empty((2 if part == 'norm' else 2 * N, len(z)))
    for k in range(len(z)):
        zp, zm = z.copy(), z.copy(); zp[k] += h; zm[k] -= h
        J[:, k] = (Rfun(zp) - Rfun(zm)) / (2 * h)
    return J


def solve(z0, Rt):
    from scipy.optimize import minimize
    cons = [{"type": "eq", "fun": model.constraints, "jac": model.constraint_jac},
            {"type": "eq", "fun": lambda z: Rfun(z) - Rt, "jac": Rjac}]
    return minimize(lambda z: model.energy(z), z0, jac=True, method="SLSQP", constraints=cons, options=dict(maxiter=1000, ftol=1e-13))


R1 = np.concatenate(Rprime(1.0)); print(f"{tag} Ng={ng} U/t={U}: |R(phi) - R(file)| at s=1: {np.abs(Rfun(z_sol) - R1).max():.1e}", flush=True)
t0 = time.time()
HMIN = 1e-3
os.makedirs(f"{base}/s_1.0000", exist_ok=True); np.save(f"{base}/s_1.0000/z.npy", z_sol)
points = {1.0: (model.energy(z_sol, grad=False), z_sol)}


def walk(direction, limit):
    s_cur, z, h = 1.0, z_sol, ds
    while (limit - s_cur) * direction > 1e-9:
        s_new = s_cur + direction * min(h, abs(limit - s_cur)); s_new = round(s_new, 6)
        res = solve(z, np.concatenate(Rprime(s_new)))
        viol = max(np.abs(model.constraints(res.x)).max(), np.abs(Rfun(res.x) - np.concatenate(Rprime(s_new))).max())
        ok = bool(res.success) and viol < 1e-7
        print(f"  s={s_new:.4f} (step {abs(s_new - s_cur):.4f}) from s={s_cur:.4f}: E_var={res.fun:.8f} viol={viol:.1e} nit={res.nit} success={res.success} -> {'accepted' if ok else 'rejected, halving the step'}", flush=True)
        if ok:
            pd = f"{base}/s_{s_new:.4f}"; os.makedirs(pd, exist_ok=True); np.save(pd + "/z.npy", res.x)
            points[s_new] = (res.fun, res.x); s_cur, z, h = s_new, res.x, min(1.5 * h, ds)
        else:
            h /= 2
            if h < HMIN:
                print(f"  walk stopped at s={s_cur:.4f} (step below {HMIN})", flush=True); return s_cur
    return s_cur


reach = (walk(+1, smax), walk(-1, smin))
for s_ in sorted(points, reverse=True):
    E, z = points[s_]; pd = f"{base}/s_{s_:.4f}"
    r = G.build_psi(pd, U, dV, Ng=ng, verbose=False); a = G.analyse(r["Psi"], U, dV)
    Rt = np.concatenate(Rprime(s_)); dR = np.linalg.norm(r["info"][0]["Rchk"]) ** 2 - Rt[0]
    open(txt, "a").write(" ".join(f"{x:.10g}" for x in [s_, E, 0.0, np.sqrt(Rt[0]), a["pion"], a["E"], a["ovl"], dR]) + "\n")
print(f"done ({time.time()-t0:.0f}s); {len(points)} points, reached s from {reach[1]:.4f} to {reach[0]:.4f}", flush=True)
